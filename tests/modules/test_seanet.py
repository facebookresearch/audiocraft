# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import pytest
import torch

from audiocraft.modules.seanet import SEANetEncoder, SEANetDecoder, SEANetResnetBlock
from audiocraft.modules import StreamableConv1d, StreamableConvTranspose1d


class TestSEANetModel:

    def test_base(self):
        encoder = SEANetEncoder()
        decoder = SEANetDecoder()

        x = torch.randn(1, 1, 24000)
        z = encoder(x)
        assert list(z.shape) == [1, 128, 75], z.shape
        y = decoder(z)
        assert y.shape == x.shape, (x.shape, y.shape)

    def test_causal(self):
        encoder = SEANetEncoder(causal=True)
        decoder = SEANetDecoder(causal=True)
        x = torch.randn(1, 1, 24000)

        z = encoder(x)
        assert list(z.shape) == [1, 128, 75], z.shape
        y = decoder(z)
        assert y.shape == x.shape, (x.shape, y.shape)

    def test_conv_skip_connection(self):
        encoder = SEANetEncoder(true_skip=False)
        decoder = SEANetDecoder(true_skip=False)

        x = torch.randn(1, 1, 24000)
        z = encoder(x)
        assert list(z.shape) == [1, 128, 75], z.shape
        y = decoder(z)
        assert y.shape == x.shape, (x.shape, y.shape)

    def test_seanet_encoder_decoder_final_act(self):
        encoder = SEANetEncoder(true_skip=False)
        decoder = SEANetDecoder(true_skip=False, final_activation='Tanh')

        x = torch.randn(1, 1, 24000)
        z = encoder(x)
        assert list(z.shape) == [1, 128, 75], z.shape
        y = decoder(z)
        assert y.shape == x.shape, (x.shape, y.shape)

    def _check_encoder_blocks_norm(self, encoder: SEANetEncoder, n_disable_blocks: int, norm: str):
        n_blocks = 0
        for layer in encoder.model:
            if isinstance(layer, StreamableConv1d):
                n_blocks += 1
                assert layer.conv.norm_type == 'none' if n_blocks <= n_disable_blocks else norm
            elif isinstance(layer, SEANetResnetBlock):
                for resnet_layer in layer.block:
                    if isinstance(resnet_layer, StreamableConv1d):
                        # here we add + 1 to n_blocks as we increment n_blocks just after the block
                        assert resnet_layer.conv.norm_type == 'none' if (n_blocks + 1) <= n_disable_blocks else norm

    def test_encoder_disable_norm(self):
        n_residuals = [0, 1, 3]
        disable_blocks = [0, 1, 2, 3, 4, 5, 6]
        norms = ['weight_norm', 'none']
        for n_res, disable_blocks, norm in product(n_residuals, disable_blocks, norms):
            encoder = SEANetEncoder(n_residual_layers=n_res, norm=norm,
                                    disable_norm_outer_blocks=disable_blocks)
            self._check_encoder_blocks_norm(encoder, disable_blocks, norm)

    def _check_decoder_blocks_norm(self, decoder: SEANetDecoder, n_disable_blocks: int, norm: str):
        n_blocks = 0
        for layer in decoder.model:
            if isinstance(layer, StreamableConv1d):
                n_blocks += 1
                assert layer.conv.norm_type == 'none' if (decoder.n_blocks - n_blocks) < n_disable_blocks else norm
            elif isinstance(layer, StreamableConvTranspose1d):
                n_blocks += 1
                assert layer.convtr.norm_type == 'none' if (decoder.n_blocks - n_blocks) < n_disable_blocks else norm
            elif isinstance(layer, SEANetResnetBlock):
                for resnet_layer in layer.block:
                    if isinstance(resnet_layer, StreamableConv1d):
                        assert resnet_layer.conv.norm_type == 'none' \
                            if (decoder.n_blocks - n_blocks) < n_disable_blocks else norm

    def test_decoder_disable_norm(self):
        n_residuals = [0, 1, 3]
        disable_blocks = [0, 1, 2, 3, 4, 5, 6]
        norms = ['weight_norm', 'none']
        for n_res, disable_blocks, norm in product(n_residuals, disable_blocks, norms):
            decoder = SEANetDecoder(n_residual_layers=n_res, norm=norm,
                                    disable_norm_outer_blocks=disable_blocks)
            self._check_decoder_blocks_norm(decoder, disable_blocks, norm)

    def test_disable_norm_raises_exception(self):
        # Invalid disable_norm_outer_blocks values raise exceptions
        with pytest.raises(AssertionError):
            SEANetEncoder(disable_norm_outer_blocks=-1)

        with pytest.raises(AssertionError):
            SEANetEncoder(ratios=[1, 1, 2, 2], disable_norm_outer_blocks=7)

        with pytest.raises(AssertionError):
            SEANetDecoder(disable_norm_outer_blocks=-1)

        with pytest.raises(AssertionError):
            SEANetDecoder(ratios=[1, 1, 2, 2], disable_norm_outer_blocks=7)
