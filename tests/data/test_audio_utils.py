# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import julius
import torch
import pytest

from audiocraft.data.audio_utils import (
    _clip_wav,
    convert_audio_channels,
    convert_audio,
    f32_pcm,
    i16_pcm,
    normalize_audio
)
from ..common_utils import get_batch_white_noise


class TestConvertAudioChannels:

    def test_convert_audio_channels_downmix(self):
        b, c, t = 2, 3, 100
        audio = get_batch_white_noise(b, c, t)
        mixed = convert_audio_channels(audio, channels=2)
        assert list(mixed.shape) == [b, 2, t]

    def test_convert_audio_channels_nochange(self):
        b, c, t = 2, 3, 100
        audio = get_batch_white_noise(b, c, t)
        mixed = convert_audio_channels(audio, channels=c)
        assert list(mixed.shape) == list(audio.shape)

    def test_convert_audio_channels_upmix(self):
        b, c, t = 2, 1, 100
        audio = get_batch_white_noise(b, c, t)
        mixed = convert_audio_channels(audio, channels=3)
        assert list(mixed.shape) == [b, 3, t]

    def test_convert_audio_channels_upmix_error(self):
        b, c, t = 2, 2, 100
        audio = get_batch_white_noise(b, c, t)
        with pytest.raises(ValueError):
            convert_audio_channels(audio, channels=3)


class TestConvertAudio:

    def test_convert_audio_channels_downmix(self):
        b, c, dur = 2, 3, 4.
        sr = 128
        audio = get_batch_white_noise(b, c, int(sr * dur))
        out = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=2)
        assert list(out.shape) == [audio.shape[0], 2, audio.shape[-1]]

    def test_convert_audio_channels_upmix(self):
        b, c, dur = 2, 1, 4.
        sr = 128
        audio = get_batch_white_noise(b, c, int(sr * dur))
        out = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=3)
        assert list(out.shape) == [audio.shape[0], 3, audio.shape[-1]]

    def test_convert_audio_upsample(self):
        b, c, dur = 2, 1, 4.
        sr = 2
        new_sr = 3
        audio = get_batch_white_noise(b, c, int(sr * dur))
        out = convert_audio(audio, from_rate=sr, to_rate=new_sr, to_channels=c)
        out_j = julius.resample.resample_frac(audio, old_sr=sr, new_sr=new_sr)
        assert torch.allclose(out, out_j)

    def test_convert_audio_resample(self):
        b, c, dur = 2, 1, 4.
        sr = 3
        new_sr = 2
        audio = get_batch_white_noise(b, c, int(sr * dur))
        out = convert_audio(audio, from_rate=sr, to_rate=new_sr, to_channels=c)
        out_j = julius.resample.resample_frac(audio, old_sr=sr, new_sr=new_sr)
        assert torch.allclose(out, out_j)

    def test_convert_pcm(self):
        b, c, dur = 2, 1, 4.
        sr = 3
        i16_audio = torch.randint(-2**15, 2**15, (b, c, int(sr * dur)), dtype=torch.int16)
        f32_audio = f32_pcm(i16_audio)
        another_i16_audio = i16_pcm(f32_audio)
        assert torch.allclose(i16_audio, another_i16_audio)


class TestNormalizeAudio:

    def test_clip_wav(self):
        b, c, dur = 2, 1, 4.
        sr = 3
        audio = 10.0 * get_batch_white_noise(b, c, int(sr * dur))
        _clip_wav(audio)
        assert audio.abs().max() <= 1

    def test_normalize_audio_clip(self):
        b, c, dur = 2, 1, 4.
        sr = 3
        audio = 10.0 * get_batch_white_noise(b, c, int(sr * dur))
        norm_audio = normalize_audio(audio, strategy='clip')
        assert norm_audio.abs().max() <= 1

    def test_normalize_audio_rms(self):
        b, c, dur = 2, 1, 4.
        sr = 3
        audio = 10.0 * get_batch_white_noise(b, c, int(sr * dur))
        norm_audio = normalize_audio(audio, strategy='rms')
        assert norm_audio.abs().max() <= 1

    def test_normalize_audio_peak(self):
        b, c, dur = 2, 1, 4.
        sr = 3
        audio = 10.0 * get_batch_white_noise(b, c, int(sr * dur))
        norm_audio = normalize_audio(audio, strategy='peak')
        assert norm_audio.abs().max() <= 1
