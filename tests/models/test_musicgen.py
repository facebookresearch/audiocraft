# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from audiocraft.models import MusicGen


class TestSEANetModel:
    def get_musicgen(self):
        mg = MusicGen.get_pretrained(name='debug', device='cpu')
        mg.set_generation_params(duration=2.0)
        return mg

    def test_base(self):
        mg = self.get_musicgen()
        assert mg.frame_rate == 25
        assert mg.sample_rate == 32000
        assert mg.audio_channels == 1

    def test_generate_unconditional(self):
        mg = self.get_musicgen()
        wav = mg.generate_unconditional(3)
        assert list(wav.shape) == [3, 1, 64000]

    def test_generate_continuation(self):
        mg = self.get_musicgen()
        prompt = torch.randn(3, 1, 32000)
        wav = mg.generate_continuation(prompt, 32000)
        assert list(wav.shape) == [3, 1, 64000]

        prompt = torch.randn(2, 1, 32000)
        wav = mg.generate_continuation(
            prompt, 32000, ['youpi', 'lapin dort'])
        assert list(wav.shape) == [2, 1, 64000]

        prompt = torch.randn(2, 1, 32000)
        with pytest.raises(AssertionError):
            wav = mg.generate_continuation(
                prompt, 32000, ['youpi', 'lapin dort', 'one too many'])

    def test_generate(self):
        mg = self.get_musicgen()
        wav = mg.generate(
            ['youpi', 'lapin dort'])
        assert list(wav.shape) == [2, 1, 64000]
