import torch.nn as nn
import torch
import math


# pos_encoding
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3):
    x = .5 if exponential else 1
    cpe = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        # pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe is None:
        w_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        w_pos = torch.empty((q_len, 1))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == 'zeros':
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        w_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': w_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': w_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': w_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': w_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': w_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(w_pos, requires_grad=learn_pe)


class ModelPTST(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, n_heads):
        super(ModelPTST, self).__init__()
        # model
        decomposition = False
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = SeriesDecomp(5)
            self.model_trend = PatchTSTBackbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                                 patch_len=patch_len, stride=8, n_layers=2, d_model=128,
                                                 n_heads=n_heads, d_k = None, d_v = None, d_ff=256, dropout=0.2,
                                                 pe='zeros', learn_pe=False, fc_dropout=0.2, padding_patch='end',
                                                 pretrain_head=False, head_type='flatten', individual=True)
            self.model_res = PatchTSTBackbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                               patch_len=patch_len, stride=8, n_layers=2, d_model=128,
                                               n_heads=n_heads, d_k = None, d_v = None, d_ff=256, dropout=0.2,
                                               pe='zeros', learn_pe=False, fc_dropout=0.2, padding_patch='end',
                                               pretrain_head=False, head_type='flatten', individual=True)
        else:
            self.model_res = PatchTSTBackbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                              patch_len=patch_len, stride=8, n_layers=2, d_model=128,
                                              n_heads=n_heads, d_k = None, d_v = None, d_ff=256, dropout=0.2,
                                              pe='zeros', learn_pe=False, fc_dropout=0.2, padding_patch='end',
                                              pretrain_head=False, head_type='flatten', individual=True)

    def forward(self, x):  # x: [Batch, Channel, Input length]
        x = x.permute(0, 2, 1) # [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2,
                                                                                 1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model_res(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1) # [Batch, Channel, Input length]
        return x


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class PatchTSTBackbone(nn.Module):
    def __init__(self, c_in: int, context_window: int, target_window: int, patch_len: int, stride: int,
                 n_layers = 3, d_model=128, n_heads=16, d_k = None, d_v = None, d_ff: int = 256, dropout: float = 0.,
                 pe: str = 'zeros', learn_pe = False, fc_dropout: float = 0., padding_patch=None,
                 pretrain_head = False, head_type='flatten', individual=True):
        super(PatchTSTBackbone, self).__init__()
        # RevIn
        self.rev_in = RevIN
        if self.rev_in: self.rev_in_layer = RevIN(c_in, affine=True, subtract_last=False)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(patch_num=patch_num, patch_len=patch_len, n_layers=n_layers, d_model=d_model,
                                    n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm='BatchNorm', dropout=dropout,
                                    pe=pe, learn_pe=learn_pe)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in,
                                                  fc_dropout)  # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = FlattenHead(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=0)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.rev_in:
            z = z.permute(0, 2, 1)
            z = self.rev_in_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x target_window]

        # denorm
        if self.rev_in:
            z = z.permute(0, 2, 1)
            z = self.rev_in_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z

    @staticmethod
    def create_pretrain_head(head_nf, n_vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                             nn.Conv1d(head_nf, n_vars, 1)
                             )


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


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, patch_num, patch_len, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None, d_ff=256,
                 norm='BatchNorm', dropout=0., pe='zeros', learn_pe=False):
        super(TSTiEncoder, self).__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dropout=dropout,
                                  n_layers=n_layers)

    def forward(self, x):  # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]
        return z


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', dropout=0., n_layers=1):
        super(TSTEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, dropout=dropout, ) for i
             in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers: output = mod(output)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, norm='BatchNorm', dropout=0., bias=True):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src):
        # Multi-Head attention
        src2, attn = self.self_attn(src, src, src)
        # Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        src = self.norm_attn(src)

        # Feed-forward sublayer
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        src = self.norm_ffn(src)
        return src


class Transpose(nn.Module):
    def __init__(self, *dims):
        super(Transpose, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, qkv_bias=True):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super(_MultiHeadAttention, self).__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model))

    def forward(self, q, k, v):
        bs = q.size(0)
        if k is None: k = q
        if v is None: v = q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(k).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(v).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        output, attn_weights = self.sdp_attn(q_s, k_s, v_s)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)
        return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.):
        super(_ScaledDotProductAttention, self).__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q, k, v):
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale    # attn_scores: [bs x n_heads x max_q_len x q_len]

        # normalize the attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)# attn_weights: [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)           # output: [bs x n_heads x max_q_len x d_v]
        return output, attn_weights
