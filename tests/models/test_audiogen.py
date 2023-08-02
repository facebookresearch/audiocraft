# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from audiocraft.models import AudioGen


class TestAudioGenModel:
    def get_audiogen(self):
        ag = AudioGen.get_pretrained(name='debug', device='cpu')
        ag.set_generation_params(duration=2.0, extend_stride=2.)
        return ag

    def test_base(self):
        ag = self.get_audiogen()
        assert ag.frame_rate == 25
        assert ag.sample_rate == 16000
        assert ag.audio_channels == 1

    def test_generate_continuation(self):
        ag = self.get_audiogen()
        prompt = torch.randn(3, 1, 16000)
        wav = ag.generate_continuation(prompt, 16000)
        assert list(wav.shape) == [3, 1, 32000]

        prompt = torch.randn(2, 1, 16000)
        wav = ag.generate_continuation(
            prompt, 16000, ['youpi', 'lapin dort'])
        assert list(wav.shape) == [2, 1, 32000]

        prompt = torch.randn(2, 1, 16000)
        with pytest.raises(AssertionError):
            wav = ag.generate_continuation(
                prompt, 16000, ['youpi', 'lapin dort', 'one too many'])

    def test_generate(self):
        ag = self.get_audiogen()
        wav = ag.generate(
            ['youpi', 'lapin dort'])
        assert list(wav.shape) == [2, 1, 32000]

    def test_generate_long(self):
        ag = self.get_audiogen()
        ag.max_duration = 3.
        ag.set_generation_params(duration=4., extend_stride=2.)
        wav = ag.generate(
            ['youpi', 'lapin dort'])
        assert list(wav.shape) == [2, 1, 16000 * 4]
