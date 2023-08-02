# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch

from audiocraft.adversarial.discriminators import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    MultiScaleSTFTDiscriminator
)


class TestMultiPeriodDiscriminator:

    def test_mpd_discriminator(self):
        N, C, T = 2, 2, random.randrange(1, 100_000)
        t0 = torch.randn(N, C, T)
        periods = [1, 2, 3]
        mpd = MultiPeriodDiscriminator(periods=periods, in_channels=C)
        logits, fmaps = mpd(t0)

        assert len(logits) == len(periods)
        assert len(fmaps) == len(periods)
        assert all([logit.shape[0] == N and len(logit.shape) == 4 for logit in logits])
        assert all([feature.shape[0] == N for fmap in fmaps for feature in fmap])


class TestMultiScaleDiscriminator:

    def test_msd_discriminator(self):
        N, C, T = 2, 2, random.randrange(1, 100_000)
        t0 = torch.randn(N, C, T)

        scale_norms = ['weight_norm', 'weight_norm']
        msd = MultiScaleDiscriminator(scale_norms=scale_norms, in_channels=C)
        logits, fmaps = msd(t0)

        assert len(logits) == len(scale_norms)
        assert len(fmaps) == len(scale_norms)
        assert all([logit.shape[0] == N and len(logit.shape) == 3 for logit in logits])
        assert all([feature.shape[0] == N for fmap in fmaps for feature in fmap])


class TestMultiScaleStftDiscriminator:

    def test_msstftd_discriminator(self):
        N, C, T = 2, 2, random.randrange(1, 100_000)
        t0 = torch.randn(N, C, T)

        n_filters = 4
        n_ffts = [128, 256, 64]
        hop_lengths = [32, 64, 16]
        win_lengths = [128, 256, 64]

        msstftd = MultiScaleSTFTDiscriminator(filters=n_filters, n_ffts=n_ffts, hop_lengths=hop_lengths,
                                              win_lengths=win_lengths, in_channels=C)
        logits, fmaps = msstftd(t0)

        assert len(logits) == len(n_ffts)
        assert len(fmaps) == len(n_ffts)
        assert all([logit.shape[0] == N and len(logit.shape) == 4 for logit in logits])
        assert all([feature.shape[0] == N for fmap in fmaps for feature in fmap])
