from unittest.mock import patch

import torch.nn as nn
import torch
from typing import Union
import math


class PatchTSMixerForPrediction(nn.Module):
    def __init__(self, context_length, in_features, out_features, prediction_length, loss = "mse"):
        super(PatchTSMixerForPrediction, self).__init__()
        num_parallel_samples = 100
        distribution_output = "student_t"
        self.loss_type = loss
        self.prediction_channel_indices = None
        self.num_parallel_samples = num_parallel_samples

        if loss == "mse":
            self.distribution_output = None
        patch_length = 8
        d_model = 8
        patch_stride = 8
        num_patch = ((context_length - patch_length) // patch_stride) + 1

        # set distribution_output
        self.model = PatchTSMixerModel(context_length, in_features, out_features, num_input_channels=in_features,
                                       patch_length=patch_length, patch_stride=patch_stride,
                                       d_model=d_model, num_patch=num_patch)
        self.head = PatchTSMixerForPredictionHead(prediction_channel_indices=None, head_dropout=0.2,
                                                  num_patches=num_patch,
                                                  d_model=d_model,
                                                  prediction_length=prediction_length,
                                                  distribution_output=self.distribution_output)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask = None,
        future_values = None,
        output_hidden_states: bool = False,
        return_loss: bool = True,
    ):
        # select loss function
        past_values = past_values.permute(0, 2, 1)
        if self.loss_type == "mse":
            loss_fn = nn.MSELoss(reduction="mean")
        elif self.loss_type == "nll":
            loss_fn = nll
        else:
            raise ValueError("Invalid loss function: Allowed values: mse and nll")

        # model output
        last_hidden_state, hidden_states, patch_input, mask, loc, scale = self.model(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states
        )
        # head -> predict value (batch_size, prediction_length, num_channels)
        y_hat = self.head(last_hidden_state)

        loss_val = None
        if self.prediction_channel_indices is not None:
            y_hat = y_hat[..., self.prediction_channel_indices]
            loc = loc[..., self.prediction_channel_indices]
            scale = scale[..., self.prediction_channel_indices]
            future_values = future_values[..., self.prediction_channel_indices] if future_values is not None else None

        if self.distribution_output:
            distribution = self.distribution_output.distribution(y_hat, loc=loc, scale=scale)
            if future_values is not None and return_loss:
                loss_val = loss_fn(distribution, future_values)
                loss_val = weighted_average(loss_val)
        else:
            # reverse scaling
            y_hat = y_hat * scale + loc
            if future_values is not None and return_loss:
                loss_val = loss_fn(y_hat, future_values)
        y_hat = y_hat.permute(0, 2, 1)
        # loss_val, y_hat, last_hidden_state, hidden_states, loc, scale
        return y_hat

    def generate(self, past_values: torch.Tensor, observed_mask = None):
        """
        generate prediction samples (batch_size, num_samples, prediction_length, num_input_channels)
        """
        num_parallel_samples = self.num_parallel_samples

        _, y_hat, _, _, loc, scale = self.forward(
            past_values=past_values,
            future_values=None,
            observed_mask=observed_mask,
            output_hidden_states=False,
            return_loss=False,
        )

        distribution = self.distribution_output.distribution(y_hat, loc=loc, scale=scale)
        samples = [distribution.sample() for _ in range(num_parallel_samples)]
        samples = torch.stack(samples, dim=1)
        return samples  # (batch_size, num_samples, prediction_length, num_channels)


class PatchTSMixerForPredictionHead(nn.Module):
    """Prediction Head for Forecasting
    Args:
            Configuration.
    """

    def __init__(self, prediction_channel_indices, head_dropout, num_patches, d_model, prediction_length,
                 distribution_output=None):
        super().__init__()

        self.prediction_channel_indices = prediction_channel_indices

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()

        self.dropout_layer = nn.Dropout(head_dropout)
        if distribution_output is None:
            self.base_forecast_block = nn.Linear((num_patches * d_model), prediction_length)
        else:
            self.base_forecast_block = distribution_output.get_parameter_projection(
                num_patches * d_model
            )

        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features):
        """

        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size, num_patch, d_model)` in `flatten` mode
                or `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, nvars)`.

        """

        hidden_features = self.flatten(hidden_features)  # [batch_size x n_vars x num_patch * d_model]
        hidden_features = self.dropout_layer(hidden_features)  # [batch_size x n_vars x num_patch * d_model]
        forecast = self.base_forecast_block(hidden_features)  # [batch_size x n_vars x prediction_length]
        if isinstance(forecast, tuple):
            forecast = tuple(z.transpose(-1, -2) for z in forecast)
        else:
            forecast = forecast.transpose(-1, -2)  # [batch_size x prediction_length x n_vars]

        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(z[..., self.prediction_channel_indices] for z in forecast)
            else:
                forecast = forecast[..., self.prediction_channel_indices]  # [batch_size x prediction_length x n_vars]
        return forecast


def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


def weighted_average(input_tensor: torch.Tensor, weights = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.
    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.
    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=input_tensor.dtype, device=input_tensor.device)
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


class PatchTSMixerModel(nn.Module):
    def __init__(self, context_length, in_features, out_features, num_input_channels, patch_length, patch_stride,
                 d_model, num_patch, scaling_dim=None, keepdim=None,
                 minimum_scale=None, default_scale=None, mask_input=False):

        super(PatchTSMixerModel, self).__init__()
        scaling = "std"
        self.encoder = PatchTSMixerEncoder(in_features=in_features, out_features=out_features,
                                           num_input_channels=num_input_channels,
                                           patch_length=patch_length, d_model=d_model, use_positional_encoding=False,
                                           positional_encoding_type="sincos", num_patches=num_patch, num_layers=3,
                                           mode="mix_channel", norm_mlp="LayerNorm",
                                           norm_eps=1e-05, self_attn=False,
                                           gated_attn=True, expansion_factor=2, dropout=0.2, self_attn_heads=1)

        self.patching = PatchTSMixerPatchify(context_length, patch_length=patch_length, patch_stride=patch_stride)

        if mask_input:
            self.masking = PatchTSMixerMasking(
                random_mask_ratio=0.5, channel_consistent_masking=True, mask_type=0, num_forecast_mask_patches=[2],
                unmasked_channel_indices=None, mask_value=0
            )
        else:
            self.masking = None

        if scaling == "mean":
            self.scaler = PatchTSMixerMeanScaler(scaling_dim, keepdim, minimum_scale, default_scale)
        elif scaling == "std" or scaling is True:
            self.scaler = PatchTSMixerStdScaler(scaling_dim, keepdim, minimum_scale)
        else:
            self.scaler = PatchTSMixerNOPScaler(scaling_dim, keepdim)

    def forward(self,
                past_values: torch.Tensor,
                observed_mask: torch.Tensor = None,
                output_hidden_states: bool = False):
        """
        Args:
            past_values (torch.FloatTensor): Input time series of shape (batch_size, seq_len, num_input_channels)
            observed_mask (torch.FloatTensor, optional): Same shape as past_values. 1 for observed, 0 for missing.
            output_hidden_states (bool): Whether to return hidden states from all encoder layers.

        Returns:
            Tuple:
                - last_hidden_state (Tensor): (batch_size, n_vars, num_patches, d_model)
                - hidden_states (List[Tensor] or None)
                - patch_input (Tensor): input after patchify
                - mask (Tensor or None): if masking was applied
                - loc (Tensor): scaling mean
                - scale (Tensor): scaling std or min
        """
        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)
        patched_x = self.patching(scaled_past_values)
        enc_input = patched_x
        mask = None
        if self.masking is not None:
            enc_input, mask = self.masking(patched_x)
        # Encode
        encoder_output = self.encoder(
            enc_input,
            output_hidden_states=output_hidden_states
        )
        if output_hidden_states:
            last_hidden_state, hidden_states = encoder_output
        else:
            last_hidden_state = encoder_output
            hidden_states = None
        return last_hidden_state, hidden_states, patched_x, mask, loc, scale


class PatchTSMixerEncoder(nn.Module):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        patch_length (int): Length of each patch from the input time series.
        d_model (int): Dimensionality of patch embeddings.
        use_positional_encoding (bool): Whether to apply positional encoding.
        positional_encoding_type (str): Type of positional encoding used.
        num_patches (int): Number of patches per series.
        num_layers (int): Number of MLP-Mixer layers.
        mode (str): Mixing mode.
        norm_mlp, norm_eps, self_attn, gated_attn, expansion_factor, dropout, self_attn_heads, in_features,
        out_features, num_input_channels: Architecture-related parameters passed to PatchTSMixerBlock.
    """

    def __init__(self, patch_length, d_model, use_positional_encoding,
                 positional_encoding_type, num_patches, num_layers, mode, norm_mlp, norm_eps, self_attn, gated_attn,
                 expansion_factor, dropout, self_attn_heads, in_features, out_features, num_input_channels):
        super(PatchTSMixerEncoder, self).__init__()

        self.patcher = nn.Linear(patch_length, d_model)

        if use_positional_encoding:
            self.positional_encoder = PatchTSMixerPositionalEncoding(
                use_positional_encoding, positional_encoding_type, num_patches, d_model
            )
        else:
            self.positional_encoder = None

        self.mlp_mixer_encoder = PatchTSMixerBlock(
            num_layers, mode, norm_mlp, norm_eps, d_model,
            self_attn, gated_attn, num_patches, expansion_factor,
            dropout, self_attn_heads, in_features, out_features, num_input_channels
        )

    def forward(self, past_values: torch.Tensor, output_hidden_states: bool = False):
        """
        Args:
            past_values (torch.FloatTensor of shape (batch_size, seq_length, num_input_channels)):
                Input time series data. For univariate time series, num_input_channels=1.
            output_hidden_states (bool):
                Whether to return the hidden states of each internal layer.

        Returns:
            torch.FloatTensor: Last hidden state of shape (batch_size, n_vars, num_patches, d_model)
            Optional[List[torch.FloatTensor]]: All layer hidden states (if output_hidden_states=True)
        """
        patches = self.patcher(past_values)

        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(
            patches, output_hidden_states=output_hidden_states
        )

        if output_hidden_states:
            return last_hidden_state, hidden_states
        else:
            return last_hidden_state


class PatchTSMixerPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches
    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, context_length, patch_length, patch_stride):
        super(PatchTSMixerPatchify, self).__init__()

        self.sequence_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification
        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x new_sequence_length x num_channels]
        output = past_values[:, self.sequence_start:, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


class PatchTSMixerMasking(nn.Module):
    """
    Class to perform random or forecast masking.
    Parameters:
    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
            Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points
    """
    def __init__(self, random_mask_ratio, channel_consistent_masking, mask_type, num_forecast_mask_patches,
                 unmasked_channel_indices, mask_value):
        super(PatchTSMixerMasking, self).__init__()
        self.random_mask_ratio = random_mask_ratio
        self.channel_consistent_masking = channel_consistent_masking
        self.mask_type = mask_type
        self.num_forecast_mask_patches = num_forecast_mask_patches
        self.unmasked_channel_indices = unmasked_channel_indices
        self.mask_value = mask_value
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices = sorted(self.unmasked_channel_indices)

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input
        Return:
            masked_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                Bool tensor indicating True on masked points
        """
        if self.mask_type == "random":
            masked_input, mask = random_masking(
                inputs=patch_input,
                mask_ratio=self.random_mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value
            )
        elif self.mask_type == "forecast":
            masked_input, mask = forecast_masking(
                inputs=patch_input,
                num_forecast_mask_patches=self.num_forecast_mask_patches,
                unmasked_channel_indices=self.unmasked_channel_indices,
                mask_value=self.mask_value
            )
        else:
            raise ValueError(f"Invalid mask type {self.mask_type}.")

        # mask: [bs x num_input_channels x num_patch]
        mask = mask.bool()
        return masked_input, mask


def forecast_masking(inputs: torch.Tensor, num_forecast_mask_patches: Union[list, int],
                     unmasked_channel_indices: list = None, mask_value: int = 0):
    """Forecast masking that masks the last K patches where K is from the num_forecast_mask_patches.
    If num_forecast_mask_patches is a list, samples in the batch will be randomly masked by numbers defined in the list.
    Parameters:
        inputs (`torch.Tensor`):
            Input of shape `(bs, num_channels, num_patch, patch_length)`
        num_forecast_mask_patches (`list`):
            Number of patches to be masked at the end of each batch sample. e.g. 4 or [3, 5].
        unmasked_channel_indices (`list`, *optional*):
            Indices of channels that are not masked.
        mask_value (`int`, *optional*, defaults to 0):
            Values in the masked patches will be filled by `mask_value`.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as inputs Tensor and Mask tensor of shape `(bs,
        num_channels , num_patch)` or `(bs, tsg1, tsg2, num_channels, num_patch)`
    """
    if isinstance(num_forecast_mask_patches, int):
        num_forecast_mask_patches = [num_forecast_mask_patches]
    forecast_mask_ratios = [1 for _ in num_forecast_mask_patches]

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)

    t_list = []
    total_length = 0
    total_ratio = sum(forecast_mask_ratios)

    for patch_length, ratio in zip(num_forecast_mask_patches, forecast_mask_ratios):
        if patch_length <= 0 or patch_length >= sequence_length:
            raise ValueError(
                f"num_forecast_mask_patches {patch_length} should be greater than 0 and less than total patches."
            )
        temp_len = int(batch_size * ratio / total_ratio)
        t_list.append([patch_length, ratio, temp_len])
        total_length += temp_len

    t_list = sorted(t_list, key=lambda x: x[2])

    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)
    batch1 = 0
    for patch_len, _, temp_len in t_list:
        batch2 = batch1 + temp_len
        mask[batch1:batch2, :, -patch_len:] = 1
        batch1 = batch2

    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]

    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patch x patch_len]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]

def random_masking(inputs: torch.Tensor, mask_ratio: float, unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False, mask_value: int = 0):
    """random_masking: Mask the input considering the control variables.
    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            The input tensor to mask.
        mask_ratio (`float`):
            Masking ratio applied to mask the input data during random pretraining. It is the number between 0 and 1.
        unmasked_channel_indices (list, *optional*):
            Indices of channels that will not be masked.
        channel_consistent_masking (bool, *optional*, defaults to `False`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        mask_value (int, *optional*, defaults to 0):
            Define the value of masked patches for pretraining.
    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as input Tensor and mask tensor of shape [bs x c x
        n]
    """
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"Mask ratio {mask_ratio} has to be between 0 and 1.")

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    len_keep = int(sequence_length * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(batch_size, 1, sequence_length, device=device)  # noise in [0, 1], bs x 1 x  L
        noise = noise.repeat(1, num_channels, 1)  # bs x num_channels x time
    else:
        # noise in [0, 1], bs x num_channels x L
        noise = torch.rand(batch_size, num_channels, sequence_length, device=device)

    # mask: [bs x num_channels x num_patch]
    mask = torch.ones(batch_size, num_channels, sequence_length, device=device)
    mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


class PatchTSMixerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, scaling_dim, keepdim, minimum_scale, default_scale):
        super(PatchTSMixerMeanScaler, self).__init__()
        self.dim = 1 if scaling_dim is None else scaling_dim
        self.keepdim = True if keepdim is None else keepdim
        self.minimum_scale = 1e-10 if minimum_scale is None else minimum_scale
        self.default_scale = None if default_scale is None else default_scale

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor):
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


class PatchTSMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, scaling_dim, keepdim, minimum_scale):
        super(PatchTSMixerStdScaler, self).__init__()
        self.dim = 1 if scaling_dim is None else scaling_dim
        self.keepdim = True if keepdim is None else keepdim
        self.minimum_scale = 1e-5 if minimum_scale is None else minimum_scale

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor):
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class PatchTSMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, scaling_dim, keepdim):
        super(PatchTSMixerNOPScaler, self).__init__()
        self.dim = 1 if scaling_dim is None else scaling_dim
        self.keepdim = True if keepdim is None else keepdim

    def forward(self, data: torch.Tensor):
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


class PatchTSMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """
    def __init__(self, use_positional_encoding, positional_encoding_type, num_patches, d_model):
        super(PatchTSMixerPositionalEncoding, self).__init__()
        # positional encoding: [num_patches x d_model]
        if use_positional_encoding:
            self.position_enc = self._init_pe(positional_encoding_type, num_patches, d_model)
        else:
            self.position_enc = nn.Parameter(torch.zeros(num_patches, d_model))

    @staticmethod
    def _init_pe(positional_encoding_type, num_patches, d_model) -> nn.Parameter:
        # Positional encoding
        if positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(num_patches, d_model), requires_grad=True)
        elif positional_encoding_type == "sincos":
            position_enc = torch.zeros(num_patches, d_model)
            position = torch.arange(0, num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        # hidden_state: [bs x num_channels x num_patches x d_model]
        hidden_state = patch_input + self.position_enc
        return hidden_state


class PatchTSMixerBlock(nn.Module):
    """The main computing framework of the `PatchTSMixer` model.
    Args:
            Configuration.
    """
    def __init__(self, num_layers, mode, norm_mlp, norm_eps, d_model, self_attn, gated_attn, num_patches,
                 expansion_factor, dropout,
                 self_attn_heads, in_features, out_features, num_input_channels):
        super(PatchTSMixerBlock, self).__init__()
        num_layers = num_layers
        self.mixers = nn.ModuleList(
            [PatchTSMixerLayer(mode, norm_mlp, norm_eps, d_model, self_attn, gated_attn, num_patches,
                               expansion_factor, dropout,
                               self_attn_heads, in_features, out_features, num_input_channels) for _ in
             range(num_layers)])

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.
        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []
        embedding = hidden_state
        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)
        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.
    Args:
            Configuration.
    """

    def __init__(self, mode, norm_mlp, norm_eps, d_model, self_attn, gated_attn, num_patches,
                 expansion_factor, dropout,
                 self_attn_heads, in_features, out_features, num_input_channels):
        super(PatchTSMixerLayer, self).__init__()
        self.patch_mixer = PatchMixerBlock(norm_mlp, norm_eps, d_model, self_attn, gated_attn, num_patches,
                                           expansion_factor, dropout,
                                           self_attn_heads)
        self.feature_mixer = FeatureMixerBlock(norm_mlp, norm_eps, d_model, gated_attn, in_features, out_features,
                                               expansion_factor, dropout)
        self.mode = mode
        if mode == "mix_channel":
            self.channel_feature_mixer = PatchTSMixerChannelFeatureMixerBlock(norm_mlp, norm_eps, d_model, gated_attn,
                                                                              in_features, out_features,
                                                                              expansion_factor, dropout,
                                                                              num_input_channels)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.
        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)
        hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_patches x d_model)
        return hidden


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.
    Args:
            Configuration.
    """

    def __init__(self, norm_mlp, norm_eps, d_model, self_attn, gated_attn, num_patches, expansion_factor, dropout,
                 self_attn_heads):
        super(PatchMixerBlock, self).__init__()

        self.norm = PatchTSMixerNormLayer(norm_mlp, norm_eps, d_model)
        self.self_attn = self_attn
        self.gated_attn = gated_attn
        self.mlp = PatchTSMixerMLP(
            in_features=num_patches,
            out_features=num_patches,
            expansion_factor=expansion_factor, dropout=dropout
        )
        if gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=num_patches, out_size=num_patches)
        if self_attn:
            self.self_attn_layer = PatchTSMixerAttention(
                embed_dim=d_model,
                num_heads=self_attn_heads,
                dropout=dropout,
            )
            self.norm_attn = PatchTSMixerNormLayer(norm_mlp, norm_eps, d_model)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.
        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state
        x_attn = None
        hidden_state = self.norm(hidden_state)
        if self.self_attn:
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)

            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)
        # Transpose so that num_patches is the last dimension
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        # Transpose back
        hidden_state = hidden_state.transpose(2, 3)

        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        out = hidden_state + residual
        return out


class PatchTSMixerNormLayer(nn.Module):
    def __init__(self, norm_mlp, norm_eps, d_model):
        super(PatchTSMixerNormLayer, self).__init__()
        self.norm_mlp = norm_mlp

        if "batch" in norm_mlp.lower():
            self.norm = PatchTSMixerBatchNorm(d_model, norm_eps)
        else:
            self.norm = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            # reshape the data
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )  # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            inputs_reshaped = self.norm(inputs_reshaped)

            # put back data to the original shape
            inputs = torch.reshape(inputs_reshaped, inputs.shape)
        else:
            inputs = self.norm(inputs)

        return inputs


class PatchTSMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, d_model, norm_eps):
        super(PatchTSMixerBatchNorm, self).__init__()
        self.batchnorm = nn.BatchNorm1d(d_model, eps=norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, expansion_factor, dropout):
        super(PatchTSMixerMLP, self).__init__()
        num_hidden = in_features * expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        x = self.fc1(inputs)
        inputs = self.dropout1(nn.functional.gelu(x))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.
    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """
    def __init__(self, in_size: int, out_size: int):
        super(PatchTSMixerGatedAttention, self).__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class PatchTSMixerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout = 0.0,
        is_decoder = False,
        bias: bool = True,
        is_causal = False):
        super(PatchTSMixerAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self,
        hidden_states: torch.Tensor,
        key_value_states = None,
        past_key_value = None,
        attention_mask = None,
        layer_head_mask = None,
        output_attentions = False):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.
    Args:
            Configuration.
    """
    def __init__(self, norm_mlp, norm_eps, d_model, gated_attn, in_features, out_features, expansion_factor, dropout):
        super(FeatureMixerBlock, self).__init__()
        self.norm = PatchTSMixerNormLayer(norm_mlp, norm_eps, d_model)
        self.gated_attn = gated_attn
        self.mlp = PatchTSMixerMLP(d_model, d_model, expansion_factor, dropout)
        if gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=d_model, out_size=d_model)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.
        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        out = hidden + residual
        return out


class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.
    Args:
            Configuration.
    """
    def __init__(self, norm_mlp, norm_eps, d_model, gated_attn, in_features, out_features, expansion_factor, dropout, num_input_channels):
        super(PatchTSMixerChannelFeatureMixerBlock, self).__init__()

        self.norm = PatchTSMixerNormLayer(norm_mlp, norm_eps, d_model)
        self.gated_attn = gated_attn
        self.mlp = PatchTSMixerMLP(in_features, out_features, expansion_factor, dropout)
        if gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(
                in_size=num_input_channels, out_size=num_input_channels
            )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gating_block(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        out = inputs + residual
        return out

