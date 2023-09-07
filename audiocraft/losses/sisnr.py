# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F


def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "data should be contiguous"
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(-1, True)


def _norm2(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(-1, True)


class SISNR(nn.Module):
    """SISNR loss.

    Input should be [B, C, T], output is scalar.

    ..Warning:: This function returns the opposite of the SI-SNR (e.g. `-1 * regular_SI_SNR`).
        Consequently, lower scores are better in terms of reconstruction quality,
        in particular, it should be negative if training goes well. This done this way so
        that this module can also be used as a loss function for training model.

    Args:
        sample_rate (int): Sample rate.
        segment (float or None): Evaluate on chunks of that many seconds. If None, evaluate on
            entire audio only.
        overlap (float): Overlap between chunks, i.e. 0.5 = 50 % overlap.
        epsilon (float): Epsilon value for numerical stability.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: tp.Optional[float] = 20,
        overlap: float = 0.5,
        epsilon: float = torch.finfo(torch.float32).eps,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.epsilon = epsilon

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape
        assert ref_sig.shape == out_sig.shape

        if self.segment is None:
            frame = T
            stride = T
        else:
            frame = int(self.segment * self.sample_rate)
            stride = int(frame * (1 - self.overlap))

        epsilon = self.epsilon * frame  # make epsilon prop to frame size.

        gt = _unfold(ref_sig, frame, stride)
        est = _unfold(out_sig, frame, stride)
        if self.segment is None:
            assert gt.shape[-1] == 1

        gt = _center(gt)
        est = _center(est)
        dot = torch.einsum("bcft,bcft->bcf", gt, est)

        proj = dot[:, :, :, None] * gt / (epsilon + _norm2(gt))
        noise = est - proj

        sisnr = 10 * (
            torch.log10(epsilon + _norm2(proj)) - torch.log10(epsilon + _norm2(noise))
        )
        return -1 * sisnr[..., 0].mean()
