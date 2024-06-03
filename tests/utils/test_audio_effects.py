# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from omegaconf import OmegaConf

from audiocraft.utils.audio_effects import AudioEffects, get_audio_effects, select_audio_effects

from ..common_utils import get_batch_white_noise


class TestAudioEffect:
    SR = 16_000

    @pytest.fixture(autouse=True)
    def audio_effects(self):
        cfg = {
            "audio_effects": {
                "speed": {
                    "sample_rate": self.SR,
                    "speed_range": [0.8, 1.2]
                },
                "updownresample": {
                    "sample_rate": self.SR,
                    "intermediate_freq": 32_000,
                },
                "echo": {
                    "sample_rate": self.SR,
                    "volume_range": [0.1, 0.5],
                },
                "random_noise": {
                    "noise_std": 0.001,
                },
                "pink_noise": {
                    "noise_std": 0.01,
                },
                "lowpass_filter": {
                    "sample_rate": self.SR,
                    "cutoff_freq": 5_000,
                },
                "highpass_filter": {
                    "sample_rate": self.SR,
                    "cutoff_freq": 500,
                },
                "bandpass_filter": {
                    "sample_rate": self.SR,
                    "cutoff_freq_low": 300,
                    "cutoff_freq_high": 8_000,
                },
                "smooth": {
                    "window_size_range": [2, 10],
                },
                "boost_audio": {
                    "amount": 20,
                },
                "duck_audio": {
                    "amount": 20,
                },
                "mp3_compression": {
                    "sample_rate": self.SR,
                    "bitrate": "128k",
                },
                "aac_compression": {
                    "sample_rate": self.SR,
                    "bitrate": "128k",
                    "lowpass_freq": None,
                }
            }
        }
        weights = {
            "speed": 2.0,
            "updownresample": 0.4,
            "echo": 1.0,
            "random_noise": 3.0,
            "pink_noise": 0.5,
            "lowpass_filter": 4.0,
            "highpass_filter": 5.0,
            "bandpass_filter": 6.0,
            "smooth": 1.0,
        }
        return get_audio_effects(OmegaConf.structured(cfg)), weights

    def test_select_empty_effects(self):
        effects = select_audio_effects({})
        assert "identity" in effects and effects["identity"] == AudioEffects.identity

    def test_select_wrong_strategy(self):
        with pytest.raises(ValueError):
            _ = select_audio_effects(
                audio_effects={},
                mode="some invalid mode"
            )

    def test_selection(self, audio_effects):
        effect_cfg, weights = audio_effects
        effects = select_audio_effects(
            audio_effects=effect_cfg,
            weights=weights,
            mode="weighted"
        )
        b, c, t = 2, 4, 32000
        audio = get_batch_white_noise(b, c, t)
        for effect_name, effect_func in effects.items():
            modified_audio = effect_func(audio)
            # It is quite hard to unit test the content of the modified_audio though
            if effect_name == "speed":  # Speeding up audio should return in more frames
                assert modified_audio.size()[-1] > audio.size()[-1]
            else:
                assert modified_audio.size() == audio.size(), f"Wrong dimension in {effect_name}"
