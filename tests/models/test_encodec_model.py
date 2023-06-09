# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch

from audiocraft.models import EncodecModel
from audiocraft.modules import SEANetEncoder, SEANetDecoder
from audiocraft.quantization import DummyQuantizer


class TestEncodecModel:

    def _create_encodec_model(self,
                              sample_rate: int,
                              channels: int,
                              dim: int = 5,
                              n_filters: int = 3,
                              n_residual_layers: int = 1,
                              ratios: list = [5, 4, 3, 2],
                              **kwargs):
        frame_rate = np.prod(ratios)
        encoder = SEANetEncoder(channels=channels, dimension=dim, n_filters=n_filters,
                                n_residual_layers=n_residual_layers, ratios=ratios)
        decoder = SEANetDecoder(channels=channels, dimension=dim, n_filters=n_filters,
                                n_residual_layers=n_residual_layers, ratios=ratios)
        quantizer = DummyQuantizer()
        model = EncodecModel(encoder, decoder, quantizer, frame_rate=frame_rate,
                             sample_rate=sample_rate, channels=channels, **kwargs)
        return model

    def test_model(self):
        random.seed(1234)
        sample_rate = 24_000
        channels = 1
        model = self._create_encodec_model(sample_rate, channels)
        for _ in range(10):
            length = random.randrange(1, 10_000)
            x = torch.randn(2, channels, length)
            res = model(x)
            assert res.x.shape == x.shape

    def test_model_renorm(self):
        random.seed(1234)
        sample_rate = 24_000
        channels = 1
        model_nonorm = self._create_encodec_model(sample_rate, channels, renormalize=False)
        model_renorm = self._create_encodec_model(sample_rate, channels, renormalize=True)

        for _ in range(10):
            length = random.randrange(1, 10_000)
            x = torch.randn(2, channels, length)
            codes, scales = model_nonorm.encode(x)
            codes, scales = model_renorm.encode(x)
            assert scales is not None
