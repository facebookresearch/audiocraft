# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch
from audiocraft.models.multibanddiffusion import MultiBandDiffusion, DiffusionProcess
from audiocraft.models import EncodecModel, DiffusionUnet
from audiocraft.modules import SEANetEncoder, SEANetDecoder
from audiocraft.modules.diffusion_schedule import NoiseSchedule
from audiocraft.quantization import DummyQuantizer


class TestMBD:

    def _create_mbd(self,
                    sample_rate: int,
                    channels: int,
                    n_filters: int = 3,
                    n_residual_layers: int = 1,
                    ratios: list = [5, 4, 3, 2],
                    num_steps: int = 1000,
                    codec_dim: int = 128,
                    **kwargs):
        frame_rate = np.prod(ratios)
        encoder = SEANetEncoder(channels=channels, dimension=codec_dim, n_filters=n_filters,
                                n_residual_layers=n_residual_layers, ratios=ratios)
        decoder = SEANetDecoder(channels=channels, dimension=codec_dim, n_filters=n_filters,
                                n_residual_layers=n_residual_layers, ratios=ratios)
        quantizer = DummyQuantizer()
        compression_model = EncodecModel(encoder, decoder, quantizer, frame_rate=frame_rate,
                                         sample_rate=sample_rate, channels=channels, **kwargs)
        diffusion_model = DiffusionUnet(chin=channels, num_steps=num_steps, codec_dim=codec_dim)
        schedule = NoiseSchedule(device='cpu', num_steps=num_steps)
        DP = DiffusionProcess(model=diffusion_model, noise_schedule=schedule)
        mbd = MultiBandDiffusion(DPs=[DP], codec_model=compression_model)
        return mbd

    def test_model(self):
        random.seed(1234)
        sample_rate = 24_000
        channels = 1
        codec_dim = 128
        mbd = self._create_mbd(sample_rate=sample_rate, channels=channels, codec_dim=codec_dim)
        for _ in range(10):
            length = random.randrange(1, 10_000)
            x = torch.randn(2, channels, length)
            res = mbd.regenerate(x, sample_rate)
            assert res.shape == x.shape
