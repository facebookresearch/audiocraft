# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
import math
import random

import pytest
import torch
from torch import nn

from audiocraft.modules import (
    NormConv1d,
    NormConvTranspose1d,
    StreamableConv1d,
    StreamableConvTranspose1d,
    pad1d,
    unpad1d,
)


def test_get_extra_padding_for_conv1d():
    # TODO: Implement me!
    pass


def test_pad1d_zeros():
    x = torch.randn(1, 1, 20)

    xp1 = pad1d(x, (0, 5), mode='constant', value=0.)
    assert xp1.shape[-1] == 25
    xp2 = pad1d(x, (5, 5), mode='constant', value=0.)
    assert xp2.shape[-1] == 30
    xp3 = pad1d(x, (0, 0), mode='constant', value=0.)
    assert xp3.shape[-1] == 20
    xp4 = pad1d(x, (10, 30), mode='constant', value=0.)
    assert xp4.shape[-1] == 60

    with pytest.raises(AssertionError):
        pad1d(x, (-1, 0), mode='constant', value=0.)

    with pytest.raises(AssertionError):
        pad1d(x, (0, -1), mode='constant', value=0.)

    with pytest.raises(AssertionError):
        pad1d(x, (-1, -1), mode='constant', value=0.)


def test_pad1d_reflect():
    x = torch.randn(1, 1, 20)

    xp1 = pad1d(x, (0, 5), mode='reflect', value=0.)
    assert xp1.shape[-1] == 25
    xp2 = pad1d(x, (5, 5), mode='reflect', value=0.)
    assert xp2.shape[-1] == 30
    xp3 = pad1d(x, (0, 0), mode='reflect', value=0.)
    assert xp3.shape[-1] == 20
    xp4 = pad1d(x, (10, 30), mode='reflect', value=0.)
    assert xp4.shape[-1] == 60

    with pytest.raises(AssertionError):
        pad1d(x, (-1, 0), mode='reflect', value=0.)

    with pytest.raises(AssertionError):
        pad1d(x, (0, -1), mode='reflect', value=0.)

    with pytest.raises(AssertionError):
        pad1d(x, (-1, -1), mode='reflect', value=0.)


def test_unpad1d():
    x = torch.randn(1, 1, 20)

    u1 = unpad1d(x, (5, 5))
    assert u1.shape[-1] == 10
    u2 = unpad1d(x, (0, 5))
    assert u2.shape[-1] == 15
    u3 = unpad1d(x, (5, 0))
    assert u3.shape[-1] == 15
    u4 = unpad1d(x, (0, 0))
    assert u4.shape[-1] == x.shape[-1]

    with pytest.raises(AssertionError):
        unpad1d(x, (-1, 0))

    with pytest.raises(AssertionError):
        unpad1d(x, (0, -1))

    with pytest.raises(AssertionError):
        unpad1d(x, (-1, -1))


class TestNormConv1d:

    def test_norm_conv1d_modules(self):
        N, C, T = 2, 2, random.randrange(1, 100_000)
        t0 = torch.randn(N, C, T)

        C_out, kernel_size, stride = 1, 4, 1
        expected_out_length = int((T - kernel_size) / stride + 1)
        wn_conv = NormConv1d(C, 1, kernel_size=4, norm='weight_norm')
        gn_conv = NormConv1d(C, 1, kernel_size=4, norm='time_group_norm')
        nn_conv = NormConv1d(C, 1, kernel_size=4, norm='none')

        assert isinstance(wn_conv.norm, nn.Identity)
        assert isinstance(wn_conv.conv, nn.Conv1d)

        assert isinstance(gn_conv.norm, nn.GroupNorm)
        assert isinstance(gn_conv.conv, nn.Conv1d)

        assert isinstance(nn_conv.norm, nn.Identity)
        assert isinstance(nn_conv.conv, nn.Conv1d)

        for conv_layer in [wn_conv, gn_conv, nn_conv]:
            out = conv_layer(t0)
            assert isinstance(out, torch.Tensor)
            assert list(out.shape) == [N, C_out, expected_out_length]


class TestNormConvTranspose1d:

    def test_normalizations(self):
        N, C, T = 2, 2, random.randrange(1, 100_000)
        t0 = torch.randn(N, C, T)

        C_out, kernel_size, stride = 1, 4, 1
        expected_out_length = (T - 1) * stride + (kernel_size - 1) + 1

        wn_convtr = NormConvTranspose1d(C, C_out, kernel_size=kernel_size, stride=stride, norm='weight_norm')
        gn_convtr = NormConvTranspose1d(C, C_out, kernel_size=kernel_size, stride=stride, norm='time_group_norm')
        nn_convtr = NormConvTranspose1d(C, C_out, kernel_size=kernel_size, stride=stride, norm='none')

        assert isinstance(wn_convtr.norm, nn.Identity)
        assert isinstance(wn_convtr.convtr, nn.ConvTranspose1d)

        assert isinstance(gn_convtr.norm, nn.GroupNorm)
        assert isinstance(gn_convtr.convtr, nn.ConvTranspose1d)

        assert isinstance(nn_convtr.norm, nn.Identity)
        assert isinstance(nn_convtr.convtr, nn.ConvTranspose1d)

        for convtr_layer in [wn_convtr, gn_convtr, nn_convtr]:
            out = convtr_layer(t0)
            assert isinstance(out, torch.Tensor)
            assert list(out.shape) == [N, C_out, expected_out_length]


class TestStreamableConv1d:

    def get_streamable_conv1d_output_length(self, length, kernel_size, stride, dilation):
        # StreamableConv1d internally pads to make sure that the last window is full
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        n_frames = (length - kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        return ideal_length // stride

    def test_streamable_conv1d(self):
        N, C, T = 2, 2, random.randrange(1, 100_000)
        t0 = torch.randn(N, C, T)
        C_out = 1

        # conv params are [(kernel_size, stride, dilation)]
        conv_params = [(4, 1, 1), (4, 2, 1), (3, 1, 3), (10, 5, 1), (3, 2, 3)]
        for causal, (kernel_size, stride, dilation) in product([False, True], conv_params):
            expected_out_length = self.get_streamable_conv1d_output_length(T, kernel_size, stride, dilation)
            sconv = StreamableConv1d(C, C_out, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal)
            out = sconv(t0)
            assert isinstance(out, torch.Tensor)
            print(list(out.shape), [N, C_out, expected_out_length])
            assert list(out.shape) == [N, C_out, expected_out_length]


class TestStreamableConvTranspose1d:

    def get_streamable_convtr1d_output_length(self, length, kernel_size, stride):
        padding_total = (kernel_size - stride)
        return (length - 1) * stride - padding_total + (kernel_size - 1) + 1

    def test_streamable_convtr1d(self):
        N, C, T = 2, 2, random.randrange(1, 100_000)
        t0 = torch.randn(N, C, T)

        C_out = 1

        with pytest.raises(AssertionError):
            StreamableConvTranspose1d(C, C_out, kernel_size=4, causal=False, trim_right_ratio=0.5)
            StreamableConvTranspose1d(C, C_out, kernel_size=4, causal=True, trim_right_ratio=-1.)
            StreamableConvTranspose1d(C, C_out, kernel_size=4, causal=True, trim_right_ratio=2)

        # causal params are [(causal, trim_right)]
        causal_params = [(False, 1.0), (True, 1.0), (True, 0.5), (True, 0.0)]
        # conv params are [(kernel_size, stride)]
        conv_params = [(4, 1), (4, 2), (3, 1), (10, 5)]
        for ((causal, trim_right_ratio), (kernel_size, stride)) in product(causal_params, conv_params):
            expected_out_length = self.get_streamable_convtr1d_output_length(T, kernel_size, stride)
            sconvtr = StreamableConvTranspose1d(C, C_out, kernel_size=kernel_size, stride=stride,
                                                causal=causal, trim_right_ratio=trim_right_ratio)
            out = sconvtr(t0)
            assert isinstance(out, torch.Tensor)
            assert list(out.shape) == [N, C_out, expected_out_length]
