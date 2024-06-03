# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch

from audiocraft.losses import (
    MelSpectrogramL1Loss,
    MultiScaleMelSpectrogramLoss,
    MRSTFTLoss,
    SISNR,
    STFTLoss,
)
from audiocraft.losses.loudnessloss import TFLoudnessRatio
from audiocraft.losses.wmloss import WMMbLoss
from tests.common_utils.wav_utils import get_white_noise


def test_mel_l1_loss():
    N, C, T = 2, 2, random.randrange(1000, 100_000)
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    mel_l1 = MelSpectrogramL1Loss(sample_rate=22_050)
    loss = mel_l1(t1, t2)
    loss_same = mel_l1(t1, t1)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss_same, torch.Tensor)
    assert loss_same.item() == 0.0


def test_msspec_loss():
    N, C, T = 2, 2, random.randrange(1000, 100_000)
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    msspec = MultiScaleMelSpectrogramLoss(sample_rate=22_050)
    loss = msspec(t1, t2)
    loss_same = msspec(t1, t1)

    assert isinstance(loss, torch.Tensor)
    assert isinstance(loss_same, torch.Tensor)
    assert loss_same.item() == 0.0


def test_mrstft_loss():
    N, C, T = 2, 2, random.randrange(1000, 100_000)
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    mrstft = MRSTFTLoss()
    loss = mrstft(t1, t2)

    assert isinstance(loss, torch.Tensor)


def test_sisnr_loss():
    N, C, T = 2, 2, random.randrange(1000, 100_000)
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    sisnr = SISNR()
    loss = sisnr(t1, t2)

    assert isinstance(loss, torch.Tensor)


def test_stft_loss():
    N, C, T = 2, 2, random.randrange(1000, 100_000)
    t1 = torch.randn(N, C, T)
    t2 = torch.randn(N, C, T)

    mrstft = STFTLoss()
    loss = mrstft(t1, t2)

    assert isinstance(loss, torch.Tensor)


def test_wm_loss():
    N, nbits, T = 2, 16, random.randrange(1000, 100_000)
    positive = torch.randn(N, 2 + nbits, T)
    t2 = torch.randn(N, 1, T)
    message = torch.randn(N, nbits)

    wmloss = WMMbLoss(0.3, "mse")
    loss = wmloss(positive, None, t2, message)

    assert isinstance(loss, torch.Tensor)


def test_loudness_loss():
    sr = 16_000
    duration = 1.0
    wav = get_white_noise(1, int(sr * duration)).unsqueeze(0)
    tflrloss = TFLoudnessRatio(sample_rate=sr, n_bands=1)

    loss = tflrloss(wav, wav)
    assert isinstance(loss, torch.Tensor)
