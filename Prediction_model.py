import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as function
import psutil
import os
import Adaptive_encoding
import Autoformer
import PatchTST_
import HDMixer_
import time
import numpy as np
import TSMixer
from MSGNET import self_attention


def get_cpu_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB

def get_gpu_mem(device=0):
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return allocated, reserved

def reset_gpu_peak():
    torch.cuda.reset_peak_memory_stats()

def get_gpu_peak_mem(device=0):
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return peak_allocated, peak_reserved


class TimeSeriesCNN(nn.Module):
    def __init__(self, sequence_length, num_classes, compress_channel):
        """
        Args:
            # input_channels (int): Number of input channels (e.g., feature dimensions).
            sequence_length (int): Length of the time series.
            num_classes (int): Number of output classes for the prediction task.
        """
        super(TimeSeriesCNN, self).__init__()
        # self.num_splits = sequence_length // compress_period
        # self.compress_period = compress_period
        #
        # # period adaptation
        # self.adaptation1 = Adaptive_encoding.PeriodAdaptation(compress_size=compress_period)
        # self.adaptation2 = Adaptive_encoding.PeriodAdaptationOutput(compress_size=num_classes)

        # 1st layer of convolution and pooling
        self.conv1 = nn.Conv1d(in_channels=compress_channel, out_channels=compress_channel, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # 2nd layer of convolution and pooling
        self.conv2 = nn.Conv1d(in_channels=compress_channel, out_channels=compress_channel, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=compress_channel, out_channels=compress_channel, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # Fully connected layer
        flattened_size = sequence_length  # after two pooling, the length is reduced to 1/4 of the original
        self.fc1 = nn.Linear(flattened_size, 48)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(48, num_classes)

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): input shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: output shape (batch_size, num_classes).
        """
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


class TimeSeriesCNNCom(nn.Module):
    def __init__(self, sequence_length, num_classes, compress_period):
        """
        Args:
            # input_channels (int): Number of input channels (e.g., feature dimensions).
            sequence_length (int): Length of the time series.
            num_classes (int): Number of output classes for the prediction task.
        """
        super(TimeSeriesCNNCom, self).__init__()
        self.num_splits = sequence_length // compress_period
        self.compress_period = compress_period

        # period adaptation
        self.adaptation1 = Adaptive_encoding.PeriodAdaptation(compress_size=compress_period)
        self.adaptation2 = Adaptive_encoding.PeriodAdaptationOutput(compress_size=num_classes)

        # 1st layer of convolution and pooling
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # 2nd layer of convolution and pooling
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # fully connected layer
        flattened_size = sequence_length  # after two pooling, the length is reduced to 1/4 of the original
        self.fc1 = nn.Linear(flattened_size, 48)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(48, num_classes)

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): input shape (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: output shape (batch_size, num_classes).
        """
        x, _ = self.period_adaptation(x)
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        x, period_mean, period_v = self.period_adaptation_out(x)

        return x, period_mean, period_v

    def period_adaptation(self, x):
        split_results = []
        split_x_m = []
        for i in range(self.num_splits):
            split_x = x[:, :, i * self.compress_period:(i + 1) * self.compress_period]
            split_x_m, split_x_v = self.adaptation1(split_x)
            split_x = (split_x - split_x_m) / split_x_v
            split_results.append(split_x)

        # reconcatenate the processed segments
        x = torch.cat(split_results, dim=2)
        return x, split_x_m

    def period_adaptation_out(self, x):
        # split_results = []
        # split_x_m = []
        # split_x_m_s = []
        # for i in range(self.num_splits):
        #     split_x = x[:, i * int(self.compress_period/3) :(i + 1) * int(self.compress_period/3)]
        #     split_x_m, split_x_v = self.adaptation2(split_x)
        #     split_x = (split_x - split_x_m) / split_x_v
        #     split_results.append(split_x)
        #     split_x_m_s.append(split_x_m)
        #
        # # reconcatenate the processed segments
        # x = torch.cat(split_results, dim=1)
        # split_x_m_s = torch.cat(split_x_m_s, dim=1)
        split_x_m, split_x_v = self.adaptation2(x)
        x = (x - split_x_m) / split_x_v
        return x, split_x_m, split_x_v


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, encoded_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, encoded_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x


# Decoder
class Decoder(nn.Module):
    def __init__(self, encoded_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(encoded_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x


class TimeSeriesCNNWhole(nn.Module):
    def __init__(self, sequence_length, num_classes, compress_period, mode, t_pattern, channel, ablation):
        """
        Args:
            # input_channels (int): Number of input channels (e.g., number of feature dimensions).
            sequence_length (int): Length of the time series.
            num_classes (int): Number of output classes for the prediction task.
        """
        super(TimeSeriesCNNWhole, self).__init__()
        self.ablation = ablation
        self.t_pattern = t_pattern
        self.mode = mode
        # Key generation
        self.key_generation = Adaptive_encoding.CyclicVectorGeneratorD()
        self.ortho_key = nn.Parameter(self.key_generation(compress_period, self.mode), requires_grad=False)
        self.mask = nn.Parameter(torch.rand_like(self.ortho_key), requires_grad=True)
        self.de_ortho_key = nn.Parameter(torch.sum(self.ortho_key, dim=0), requires_grad=False)
        # self.register_buffer('ortho_key', self.key_generation(compress_period, self.mode))
        # self.ortho_key = self.key_generation(compress_period, self.mode)

        self.num_splits = sequence_length // compress_period
        self.compress_period = compress_period

        # period adaptation
        self.adaptation1 = Adaptive_encoding.PeriodAdaptation(compress_size=compress_period)
        self.adaptation2 = Adaptive_encoding.PeriodAdaptationOutput(compress_size=num_classes)

        # 1st layer of convolution and pooling
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # 2nd layer of convolution and pooling
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # fully connected layer
        flattened_size = sequence_length  # after two pooling, the length is reduced to 1/4 of the original
        self.fc1 = nn.Linear(flattened_size, num_classes)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(num_classes, num_classes)

        self.fc3 = nn.Linear(flattened_size, 48)
        self.fc4 = nn.Linear(48, num_classes)

        self.decode = Adaptive_encoding.ElementWiseProductLayer(input_size=num_classes, out_channel=channel)

        # plug in model
        # Auto_former
        self.auto_former = nn.ModuleList([Autoformer.EncoderLayer(
            attention=Autoformer.AutoCorrelationLayer(Autoformer.AutoCorrelation(factor=3), d_model=1,
                                                      n_heads=1), d_model=1, d_ff=48) for _ in range(2)])
        self.norm_layer = Autoformer.MyLayerNorm(1)

        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Tsi_model
        self.tst = PatchTST_.TSTEncoder(d_model=sequence_length, n_heads=1, d_ff=sequence_length, n_layers=2)

        # hd_mixer
        self.mixer = HDMixer_.HDMixer(enc_in=1, mix_time=True, mix_variable=True, mix_channel=True, q_len=1,
                                      d_model=sequence_length, dropout=0.)

        self.bn1 = nn.BatchNorm1d(num_features=1)
        self.bn2 = nn.BatchNorm1d(num_features=channel)

        # ts_mixer
        patch_length = 8
        patch_stride = 8
        distribution_output = None
        num_patch = ((sequence_length - patch_length) // patch_stride) + 1
        self.TSmixer = TSMixer.PatchTSMixerModel(context_length=sequence_length, in_features=1, out_features=1,
                                                 num_input_channels=1,
                                                 patch_length=patch_length, patch_stride=patch_stride,
                                                 d_model=8, num_patch=num_patch, scaling_dim=None,
                                                 keepdim=None,
                                                 minimum_scale=None, default_scale=None, mask_input=False)
        self.prehead = TSMixer.PatchTSMixerForPredictionHead(prediction_channel_indices=None, head_dropout=0.2,
                                                  num_patches=num_patch,
                                                  d_model=8,
                                                  prediction_length=num_classes,
                                                  distribution_output=distribution_output)
        self.pool = nn.AdaptiveAvgPool2d((1, num_classes))

        # Encoder-Decoder
        self.encoder = Encoder(channel, 1)
        self.decoder = Decoder(1, channel//2, channel)

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (torch.Tensor): input shape: (batch_size, input_channels, sequence_length).

        Returns:
            torch.Tensor: output shape: (batch_size, num_classes).
        """
        # compress
        device = next(self.parameters()).device  # obtain device of the model
        x = x.to(device)
        x_label = 0

        if self.ablation[0] == 0:
            # x = self.period_adaptation(x)
            mask_ortho_key = self.ortho_key * self.mask

            num_loops = x.shape[1]  # obtain the second dimension
            x_sum = None  # save the cumulative result
            for i in range(num_loops):
                x_temp = self.period_compress(x[:, i, :].unsqueeze(1),
                                              mask_ortho_key)  # process (batch_size, sequence_length)
                if x_sum is None:
                    x_sum = x_temp  # initialize x_sum
                else:
                    x_sum = x_sum + x_temp  # element-wise addition
            x_label = x_sum[:, :, -self.num_classes:]
            x = x_sum[:, :, 0:self.sequence_length]

            # adaptation
            x, _ = self.period_adaptation(x)
        elif self.ablation[0] == 1:
            x = self.encoder(x[:, :, 0:self.sequence_length].permute(0, 2, 1)).permute(0, 2, 1)

        # patch_tet
        if self.t_pattern == 'transformer':
            x = self.tst(x)
            x = self.bn1(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.fc4(x)
        elif self.t_pattern == 'convolution':
            x = self.conv1(x)
            x = self.relu1(x)

            x = self.conv2(x)
            x = self.relu2(x)
            x = self.bn1(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)

        elif self.t_pattern == 'linear':
            x = x.unsqueeze(1)
            x = self.mixer(x)
            x = self.bn1(x.squeeze(1))
            x = torch.flatten(x, start_dim=1)
            # x = self.relu3(x)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.fc2(x)

        elif self.t_pattern == 'mlp':
            x = x.permute(0, 2, 1)
            x, _, _, _, _ ,_ = self.TSmixer(x)
            x = self.prehead(x)
            x = x.permute(0, 2, 1)
            x = torch.flatten(x, start_dim=1)
            x = self.relu3(x)
            x = self.fc2(x)
        # original
        if self.ablation[1] == 0:
            x, period_mean, period_v = self.period_adaptation_out(x)

            # mask_de_ortho_key = self.de_ortho_key * self.mask1
            x = torch.matmul(x, self.ortho_key)
            x = x / self.de_ortho_key
            # self.dropout = nn.Dropout(p=0.1)  # 30% random drop

            # decompress
            x, x_ = self.decode(x.unsqueeze(1))
            if self.ablation[0] == 1:
                return x, x, x, x
            return x, period_mean, x_, x_label
        elif self.ablation[1] == 1:
            x = x.unsqueeze(1)
            x = self.decoder(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x, x, x, x

    def period_compress(self, x, mask_ortho_key):
        circular_vector = []
        for i in range(self.num_splits + 1):
            split_x = x[:, :, i * self.compress_period:(i + 1) * self.compress_period]
            circular_temp = torch.bmm(split_x, mask_ortho_key.unsqueeze(0).expand(split_x.size(0), -1, -1))
            circular_vector.append(circular_temp)
        x = torch.cat(circular_vector, dim=2)
        return x

    def period_decompress(self, x):
        circular_vector_de = []
        for i in range(self.num_splits):
            split_x = x[:, :, i * self.compress_period:(i + 1) * self.compress_period]
            circular_de_temp = torch.bmm(split_x, self.ortho_key.T.unsqueeze(0).expand(split_x.size(0), -1, -1))
            circular_vector_de.append(circular_de_temp)
        x = torch.cat(circular_vector_de, dim=2)
        return x

    def period_adaptation(self, x):
        split_results = []
        split_x_m = []
        for i in range(self.num_splits):
            split_x = x[:, :, i * self.compress_period:(i + 1) * self.compress_period]
            split_x_m, split_x_v = self.adaptation1(split_x)
            split_x = (split_x - split_x_m) / (split_x_v + 1e-6)
            split_results.append(split_x)

        # reconcatenate the processed segments
        x = torch.cat(split_results, dim=2)
        return x, split_x_m

    def period_adaptation_out(self, x):
        # split_results = []
        # split_x_m = []
        # split_x_m_s = []
        # for i in range(self.num_splits):
        #     split_x = x[:, i * int(self.compress_period/3) :(i + 1) * int(self.compress_period/3)]
        #     split_x_m, split_x_v = self.adaptation2(split_x)
        #     split_x = (split_x - split_x_m) / split_x_v
        #     split_results.append(split_x)
        #     split_x_m_s.append(split_x_m)
        #
        # # reconcatenate the processed segments
        # x = torch.cat(split_results, dim=1)
        # split_x_m_s = torch.cat(split_x_m_s, dim=1)
        split_x_m, split_x_v = self.adaptation2(x)
        x = (x - split_x_m) / (split_x_v + 1e-6)
        return x, split_x_m, split_x_v


def js_divergence(p, q):
    """calculate the JS divergence"""
    p = function.softmax(p, dim=-1)
    q = function.softmax(q, dim=-1)
    
    m = torch.div(torch.add(q, p), torch.tensor(2))
    return torch.add(0.5 * function.kl_div(function.log_softmax(p, dim=-1), m, reduction='batchmean'),
                     0.5 * function.kl_div(function.log_softmax(q, dim=-1), m, reduction='batchmean'))

def euclidean_loss(predictions, targets):
    return torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1)).mean()

def mse_loss(predictions, targets):
    return torch.mean((predictions - targets) ** 2)

def period_loss(predictions, targets):
    return ((torch.mean((predictions - targets), dim=-1))**2).mean()

def period_loss1(predictions, targets):
    loss_cor = torch.sqrt(torch.sum((targets - targets.mean(dim=-1).unsqueeze(1)) ** 2, dim=-1))
    return euclidean_loss(predictions, loss_cor)

class Trainer:
    def __init__(self, learning_rate, epochs, model, sample_length):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = model
        self.sample_length = sample_length

    def train(self, train_data, train_label, test_data, test_label, pattern):
        if train_data.ndim == 2:
            train_data = train_data.unsqueeze(1).float()
            test_data = test_data.unsqueeze(1).float()
        else:
            train_data = train_data.float()
            test_data = test_data.float()
        optimizer_c = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss = []
        start_train = time.time()
        self.model.train()  # set model to training mode
        for epoch in range(self.epochs):
            # forward propagation
            if pattern is None:
                outputs = self.model(train_data)
                loss = mse_loss(outputs, train_label)
            elif pattern == 'hdm_mixer':
                outputs, patch_loss = self.model(train_data)
                loss = mse_loss(outputs, train_label) + patch_loss
            # backpropagation
            optimizer_c.zero_grad()
            loss.backward()
            optimizer_c.step()

            # Print the loss for each epoch
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item():.4f}")

        end_train = time.time()
        training_time = end_train - start_train
        outputs = []
        start_test = time.time()
        self.model.eval()  # set model to evaluation mode
        if pattern is None:
            outputs = self.model(test_data)
        elif pattern == 'hdm_mixer':
            outputs, patch_loss = self.model(test_data)
        end_test = time.time()
        testing_time = end_test - start_test
        # print(outputs.shape)
        # print(test_label.shape)
        loss = mse_loss(outputs, test_label)

        # Memory calculation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        reset_gpu_peak()
        test_start_cpu = get_cpu_mem()
        test_start_gpu = get_gpu_mem()
        if pattern is None:
            outputs = self.model(test_data)
        elif pattern == 'hdm_mixer':
            outputs, patch_loss = self.model(test_data)
        test_end_cpu = get_cpu_mem()
        test_end_gpu = get_gpu_mem()
        test_peak_gpu = get_gpu_peak_mem()

        # # calculate the difference
        # cpu_mem_diff = test_end_cpu - test_start_cpu
        # gpu_allocated_diff = test_end_gpu[0] - test_start_gpu[0]
        # gpu_reserved_diff = test_end_gpu[1] - test_start_gpu[1]
        #
        # print the memory usage on test stage
        # print("=" * 50)
        # print("📊 Test Stage Memory Usage:")
        # print(f"CPU Before Test: {test_start_cpu:.2f} MB -> After: {test_end_cpu:.2f} MB")
        # print(f"GPU Allocated Before: {test_start_gpu[0]:.2f} MB -> After: {test_end_gpu[0]:.2f} MB")
        # print(f"GPU Reserved Before: {test_start_gpu[1]:.2f} MB -> After: {test_end_gpu[1]:.2f} MB")
        # print(f"🚀 GPU Peak Allocated: {test_peak_gpu[0]:.2f} MB, Reserved: {test_peak_gpu[1]:.2f} MB")
        # print("=" * 50)

        # check the file size
        torch.save(self.model.state_dict(), "temp_model.pth")
        file_size = os.path.getsize("temp_model.pth") / (1024 ** 2)  # MB
        print(f"Model file size: {file_size:.2f} MB")
        # delete temp file
        os.remove("temp_model.pth")
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return loss, outputs, [np.array(training_time).item(), np.array(testing_time).item()], [file_size,
                                                                                                total_params]


class TrainerCom:
    def __init__(self, learning_rate, epochs, model, sample_length):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = model
        self.sample_length = sample_length

    def train(self, train_data, train_label, test_data, test_label, alpha_):
        if train_data.ndim == 2:
            train_data = train_data.unsqueeze(1).float()
            test_data = test_data.unsqueeze(1).float()
        else:
            train_data = train_data.float()
            test_data = test_data.float()
        # prediction_model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.apply(self.init_weights).to(device)
        train_data = train_data.to(device)
        train_label = train_label.to(device)
        test_label = test_label.to(device)
        # optimizer_c = optim.Adam(prediction_model.parameters(), lr=self.learning_rate)
        optimizer_d = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=self.epochs)
        best_loss = float('inf')  # initially set to infinity
        best_model_state = None  # save the best model parameters
        self.model.train()  # set model to training mode
        start_train = time.time()
        for epoch in range(self.epochs):
            outputs, period_mean, period_v, period_f = self.model(train_data)
            loss_n = mse_loss(outputs, train_label)
            loss_m = period_loss(period_mean, period_f.squeeze(1))
            loss_d = euclidean_loss(torch.add(period_v[:, 0, :], period_v[:, 1, :]),
                                    -torch.add(outputs[:, 0, :], outputs[:, 1, :]))
            # SingleProp regularization
            # loss_sp_final = prediction_model.forward_singleprop(train_data, extract_stage='post_adaptation', epsilon=0.1)
            # loss_v = period_loss1(period_v, train_label)
            loss2 = loss_n + loss_m + loss_d * 0.01
            # loss2 = loss_n + loss_m
            # loss2 = loss_n
            # if loss2.item() < best_loss:
            #     best_loss = loss2.item()
            # backpropagation
            optimizer_d.zero_grad()
            loss2.backward()
            self.relative_global_clipping(self.model, rho=0.15)
            optimizer_d.step()
            # scheduler.step()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss2.item():.4f}, Loss_n: {loss_n.item():.4f},"
                  # f" Loss_n: {loss_n.item():.4f}, Loss_sp: {loss_sp_final.item():.4f},"
                  f" Loss_d: {loss_d.item():.4f}, Loss_m: {loss_m.item():.4f}")
        end_train = time.time()
        training_time = end_train - start_train
        start_test = time.time()
        self.model.eval()  # set model to evaluation mode
        outputs, period_mean, period_v, period_f = self.model(test_data)
        end_test = time.time()
        testing_time = end_test - start_test
        # print(outputs.shape)
        # print(test_label.shape)
        loss1 = mse_loss(outputs, test_label)
        loss2 = period_loss(period_mean, period_f.squeeze(1))
        loss3 = euclidean_loss(torch.add(period_v[:, 0, :], period_v[:, 1, :]),
                                -torch.add(outputs[:, 0, :], outputs[:, 1, :]))
        loss = [loss1, loss2, loss3]
        torch.save(self.model.state_dict(), "temp_model.pth")
        file_size = os.path.getsize("temp_model.pth") / (1024 ** 2)  # MB
        print(f"Model file size: {file_size:.2f} MB")
        # delete temp file
        os.remove("temp_model.pth")
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return loss, outputs, [np.array(training_time).item(), np.array(testing_time).item()], [file_size,
                                                                                                total_params]

    @staticmethod
    def min_max_norm(x):
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        return (x - x_min) / (x_max - x_min + 1e-8)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def relative_global_clipping(model, rho=0.1):
        total_norm = 0.
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # calculate the total parameter norm
        weight_norm = sum((p.data.norm(2) ** 2 for p in model.parameters())) ** 0.5

        # calculate the clipping threshold
        clip_threshold = rho * weight_norm

        # clip_coef
        clip_coef = min(1.0, clip_threshold / (total_norm + 1e-6))

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
