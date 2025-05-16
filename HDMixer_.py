import torch.nn as nn
import torch
import numpy as np
from torch.nn.functional import dropout


class Model(nn.Module):
    def __init__(self, enc_in, c_in, context_window, target_window, patch_len):
        super(Model, self).__init__()

        # model
        self.model = HDMixerBackbone(enc_in=enc_in, mix_time=True, mix_variable=True,
                                     mix_channel=True, n_layers=2, c_in=c_in,
                                     context_window=context_window, target_window=target_window,
                                     patch_len=patch_len, stride=4, d_model=128,
                                     lambda_=1e-1, r=1e-2, individual=True)

    def forward(self, x):  # x: [Batch, Channel, Input length]
        x, PaEN_Loss = self.model(x)  # x: [Batch, Channel, Input length]
        return x, PaEN_Loss


class HDMixerBackbone(nn.Module):
    def __init__(self, enc_in, mix_time, mix_variable, mix_channel, n_layers, c_in, context_window, target_window,
                 patch_len, stride, d_model, lambda_, r, individual=True):
        super(HDMixerBackbone, self).__init__()

        # RevIn
        self.revin = True
        if self.revin: self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = patch_num = context_window // self.stride
        self.patch_shift_linear = nn.Linear(context_window, self.patch_num * 3)
        self.box_coder = PointWhCoder(input_size=context_window, patch_count=self.patch_num, weights=(1., 1., 1.),
                                      pts=self.patch_len, tanh=True, wh_bias=torch.tensor(5. / 3.).sqrt().log()
                                      , deform_range=0.25)
        self.lambda_ = lambda_
        self.r = r
        # Backbone
        self.backbone = Encoder(enc_in, mix_time, mix_variable, mix_channel, patch_num=patch_num, patch_len=patch_len,
                                n_layers=n_layers, d_model=d_model)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head = FlattenHead(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=0)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        batch_size = z.shape[0]
        seq_len = z.shape[-1]
        if self.revin:
            z = z.permute(0 ,2 ,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0 ,2 ,1)

        x_lfp = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        x_lfp = x_lfp.permute(0 ,1 ,3 ,2)  # z: [bs x nvars x patch_len x patch_num]

        anchor_shift = self.patch_shift_linear(z).view(batch_size * self.n_vars, self.patch_num, 3)
        sampling_location_1d = self.box_coder(anchor_shift)  # B*C, self.patch_num,self.patch_len, 1
        add1d = torch.ones(size=(batch_size * self.n_vars, self.patch_num, self.patch_len, 1)).float().to \
            (sampling_location_1d.device)
        sampling_location_2d = torch.cat([sampling_location_1d, add1d], dim=-1)
        z = z.reshape(batch_size * self.n_vars, 1, 1, seq_len)
        patch = torch.nn.functional.grid_sample(z, sampling_location_2d, mode='bilinear', padding_mode='border',
                                                align_corners=False).squeeze(1)
        # B*C, self.patch_num,self.patch_len
        x_lep = patch.reshape(batch_size, self.n_vars, self.patch_num, self.patch_len).permute(0, 1, 3, 2)
        # [bs x nvars x patch_len x patch_num]
        PaEN_Loss = self.cal_PaEn(x_lfp, x_lep, self.r, self.lambda_)
        # model
        z = self.backbone(x_lep)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window]

        # denorm
        if self.revin:
            z = z.permute(0 ,2 ,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0 ,2 ,1)
        return z, PaEN_Loss

    def cal_PaEn(self, lfp, lep, r, lambda_):
        psi_lfp = self.cal_PSI(lfp, r)
        psi_lep = self.cal_PSI(lep, r)
        psi_diff = psi_lfp - psi_lep
        lep = lep.permute(0, 1, 3, 2)
        batch, n_vars, patch_num, patch_len = lep.shape
        lep = lep.reshape(batch * n_vars, patch_num, patch_len)
        sum_x = torch.sum(lep, dim=[-2, -1])
        PaEN_loss = torch.mean(sum_x * psi_diff) * lambda_  # update parameters with REINFORCE
        return PaEN_loss

    def cal_PSI(self, x, r):
        # [bs x nvars x patch_len x patch_num]
        x = x.permute(0, 1, 3, 2)
        batch, n_vars, patch_num, patch_len = x.shape
        x = x.reshape(batch * n_vars, patch_num, patch_len)
        # Generate all possible pairs of patch_num indices within each batch
        pairs = self.generate_pairs(patch_num)
        # Calculate absolute differences between pairs of sequences
        abs_diffs = torch.abs(x[:, pairs[:, 0], :] - x[:, pairs[:, 1], :])
        # Find the maximum absolute difference for each pair of sequences
        max_abs_diffs = torch.max(abs_diffs, dim=-1).values
        max_abs_diffs = max_abs_diffs.reshape(-1, patch_num, patch_num - 1)
        # Count the number of pairs with max absolute difference less than r
        c = torch.log(1 + torch.mean((max_abs_diffs < r).float(), dim=-1))
        psi = torch.mean(c, dim=-1)
        return psi

    @staticmethod
    def generate_pairs(n):
        pairs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    pairs.append([i, j])
        return np.array(pairs)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class PointCoder(nn.Module):
    def __init__(self, input_size, patch_count, weights=(1., 1.,1.), tanh=True):
        super(PointCoder, self).__init__()
        self.input_size = input_size
        self.patch_count = patch_count
        self.weights = weights
        #self._generate_anchor()
        self.tanh = tanh

    def _generate_anchor(self, device="cpu"):
        anchors = []
        patch_stride_x = 2. / self.patch_count
        for i in range(self.patch_count):
                x = -1+(0.5+i)*patch_stride_x
                anchors.append([x])
        anchors = torch.as_tensor(anchors)
        self.anchor = torch.as_tensor(anchors, device=device)
        #self.register_buffer("anchor", anchors)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pts, model_offset=None):
        assert model_offset is None
        self.boxes = self.decode(pts)
        return self.boxes

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel = 1./self.patch_count
        wx, wy = self.weights

        dx = torch.nn.functional.tanh(rel_codes[:, :, 0]/wx) * pixel if self.tanh else rel_codes[:, :, 0]*pixel / wx
        dy = torch.nn.functional.tanh(rel_codes[:, :, 1]/wy) * pixel if self.tanh else rel_codes[:, :, 1]*pixel / wy

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:,0].unsqueeze(0)
        ref_y = boxes[:,1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x
        pred_boxes[:, :, 1] = dy + ref_y
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)
        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor) * self.input_size


class PointWhCoder(PointCoder):
    def __init__(self, input_size, patch_count, weights=(1., 1., 1.), pts=1, tanh=True, wh_bias=None,
                 deform_range=0.25):
        super(PointWhCoder, self).__init__(input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh)
        self.patch_pixel = pts
        self.wh_bias = None
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(2) + wh_bias)
        self.deform_range = deform_range

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, boxes):
        self._generate_anchor(device=boxes.device)
        # print(boxes.shape)
        # print(self.wh_bias.shape)
        if self.wh_bias is not None:
            boxes[:, :, 1:] = boxes[:, :, 1:] + self.wh_bias
        self.boxes = self.decode(boxes)
        points = self.meshgrid(self.boxes)
        return points

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel_x = 2. / self.patch_count  # patch_count=in_size//stride 这里应该用2除而不是1除 得到pixel_x是两个patch中点的原本距离
        wx, ww1, ww2 = self.weights

        dx = torch.nn.functional.tanh(rel_codes[:, :, 0] / wx) * pixel_x / 4 if self.tanh else rel_codes[:, :,
                0] * pixel_x / wx  # 中心点不会偏移超过patch_len
        dw1 = torch.nn.functional.relu(torch.nn.functional.tanh(rel_codes[:, :,
                1] / ww1)) * pixel_x * self.deform_range + pixel_x  # 中心点左边长度在[stride,stride+1/4*stride]，右边同理
        dw2 = torch.nn.functional.relu(
            torch.nn.functional.tanh(rel_codes[:, :, 2] / ww2)) * pixel_x * self.deform_range + pixel_x  #
        # dw =

        pred_boxes = torch.zeros((rel_codes.shape[0], rel_codes.shape[1], rel_codes.shape[2] - 1)).to(rel_codes.device)

        ref_x = boxes[:, 0].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x - dw1
        pred_boxes[:, :, 1] = dx + ref_x + dw2
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes

    def meshgrid(self, boxes):
        B = boxes.shape[0]
        xs = boxes
        xs = torch.nn.functional.interpolate(xs, size=self.patch_pixel, mode='linear', align_corners=True)
        results = xs
        results = results.reshape(B, self.patch_count, self.patch_pixel, 1)
        # print((1+results[0])/2*336)
        return results


class FlattenHead(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(FlattenHead, self).__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class Encoder(nn.Module):  # i means channel-independent
    def __init__(self, enc_in, mix_time, mix_variable, mix_channel, patch_num, patch_len, n_layers=3, d_model=128):
        super(Encoder, self).__init__()
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len  #

        # Encoder
        self.encoder = HDMixer(enc_in, mix_time, mix_variable, mix_channel, q_len, d_model, n_layers=n_layers)

    def forward(self, x):   # x: [bs x nvars x patch_len x patch_num]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(x)  # z: [bs x nvars x patch_num x d_model]
        return z


class HDMixer(nn.Module):
    def __init__(self, enc_in, mix_time, mix_variable, mix_channel, q_len, d_model, n_layers=1, dropout=0.):
        super(HDMixer, self).__init__()
        self.layers = nn.ModuleList(
            [HDMixerLayer(enc_in, mix_time, mix_variable, mix_channel, q_len, d_model, dropout) for i in range(n_layers)])

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        output = src
        for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output


class HDMixerLayer(nn.Module):
    def __init__(self, enc_in, mix_time, mix_variable, mix_channel, q_len, d_model, dropout=0., bias=True,
                 activation="gelu"):
        super(HDMixerLayer, self).__init__()
        c_in = enc_in
        # Add & Norm
        # [bs x nvars x patch_num x d_model]
        # Position-wise Feed-Forward
        self.mix_time = mix_time
        self.mix_variable = mix_variable
        self.mix_channel = mix_channel
        self.patch_mixer = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2, bias=bias),
            self.get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model, bias=bias),
            nn.Dropout(dropout),
        )
        self.time_mixer = nn.Sequential(
            Transpose(2, 3), LayerNorm(q_len),
            nn.Linear(q_len, q_len * 2, bias=bias),
            self.get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(q_len * 2, q_len, bias=bias),
            nn.Dropout(dropout),
            Transpose(2, 3)
        )
        # [bs x nvars  x d_model  x patch_num] ->  [bs x nvars x patch_num x d_model]

        # [bs x nvars x patch_num x d_model]
        self.variable_mixer = nn.Sequential(
            Transpose(1, 3), LayerNorm(c_in),
            nn.Linear(c_in, c_in * 2, bias=bias),
            self.get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(c_in * 2, c_in, bias=bias),
            nn.Dropout(dropout),
            Transpose(1, 3)
        )

    def forward(self, src, prev = None, key_padding_mask = None,
                attn_mask = None):
        # [bs x nvars x patch_num x d_model]
        # print(src.shape)
        if self.mix_channel:
            u = self.patch_mixer(src) + src
        else:
            u = src
        if self.mix_time:
            v = self.time_mixer(u) + src
        else:
            v = u
        if self.mix_variable:
            w = self.variable_mixer(v) + src
        else:
            w = v
        out = w
        return out

    @staticmethod
    def get_activation_fn(activation):
        if callable(activation):
            return activation()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        #print(x.shape)
        B, M, D, N = x.shape
        #x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M,D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        #x = x.permute(0, 1, 3, 2)
        return x


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)
