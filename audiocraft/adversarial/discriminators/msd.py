# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import torch
import torch.nn as nn

from ...modules import NormConv1d
from .base import MultiDiscriminator, MultiDiscriminatorOutputType


class ScaleDiscriminator(nn.Module):
    """Waveform sub-discriminator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_sizes (Sequence[int]): Kernel sizes for first and last convolutions.
        filters (int): Number of initial filters for convolutions.
        max_filters (int): Maximum number of filters.
        downsample_scales (Sequence[int]): Scale for downsampling implemented as strided convolutions.
        inner_kernel_sizes (Sequence[int] or None): Kernel sizes for inner convolutions.
        groups (Sequence[int] or None): Groups for inner convolutions.
        strides (Sequence[int] or None): Strides for inner convolutions.
        paddings (Sequence[int] or None): Paddings for inner convolutions.
        norm (str): Normalization method.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        pad (str): Padding for initial convolution.
        pad_params (dict): Parameters to provide to the padding module.
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_sizes: tp.Sequence[int] = [5, 3],
                 filters: int = 16, max_filters: int = 1024, downsample_scales: tp.Sequence[int] = [4, 4, 4, 4],
                 inner_kernel_sizes: tp.Optional[tp.Sequence[int]] = None, groups: tp.Optional[tp.Sequence[int]] = None,
                 strides: tp.Optional[tp.Sequence[int]] = None, paddings: tp.Optional[tp.Sequence[int]] = None,
                 norm: str = 'weight_norm', activation: str = 'LeakyReLU',
                 activation_params: dict = {'negative_slope': 0.2}, pad: str = 'ReflectionPad1d',
                 pad_params: dict = {}):
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1
        assert (inner_kernel_sizes is None or len(inner_kernel_sizes) == len(downsample_scales))
        assert (groups is None or len(groups) == len(downsample_scales))
        assert (strides is None or len(strides) == len(downsample_scales))
        assert (paddings is None or len(paddings) == len(downsample_scales))
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                NormConv1d(in_channels, filters, kernel_size=np.prod(kernel_sizes), stride=1, norm=norm)
            )
        )

        in_chs = filters
        for i, downsample_scale in enumerate(downsample_scales):
            out_chs = min(in_chs * downsample_scale, max_filters)
            default_kernel_size = downsample_scale * 10 + 1
            default_stride = downsample_scale
            default_padding = (default_kernel_size - 1) // 2
            default_groups = in_chs // 4
            self.convs.append(
                NormConv1d(in_chs, out_chs,
                           kernel_size=inner_kernel_sizes[i] if inner_kernel_sizes else default_kernel_size,
                           stride=strides[i] if strides else default_stride,
                           groups=groups[i] if groups else default_groups,
                           padding=paddings[i] if paddings else default_padding,
                           norm=norm))
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_filters)
        self.convs.append(NormConv1d(in_chs, out_chs, kernel_size=kernel_sizes[0], stride=1,
                                     padding=(kernel_sizes[0] - 1) // 2, norm=norm))
        self.conv_post = NormConv1d(out_chs, out_channels, kernel_size=kernel_sizes[1], stride=1,
                                    padding=(kernel_sizes[1] - 1) // 2, norm=norm)

    def forward(self, x: torch.Tensor):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = self.activation(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        # x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(MultiDiscriminator):
    """Multi-Scale (MSD) Discriminator,

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample_factor (int): Downsampling factor between the different scales.
        scale_norms (Sequence[str]): Normalization for each sub-discriminator.
        **kwargs: Additional args for ScaleDiscriminator.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, downsample_factor: int = 2,
                 scale_norms: tp.Sequence[str] = ['weight_norm', 'weight_norm', 'weight_norm'], **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(in_channels, out_channels, norm=norm, **kwargs) for norm in scale_norms
        ])
        self.downsample = nn.AvgPool1d(downsample_factor * 2, downsample_factor, padding=downsample_factor)

    @property
    def num_discriminators(self):
        return len(self.discriminators)

    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        logits = []
        fmaps = []
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                self.downsample(x)
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
