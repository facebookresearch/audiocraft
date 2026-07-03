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

    def test_normalize_audio_loudness(self):
        # Loudness normalization requires a single sample (not a batch) because
        # torchaudio.transforms.Loudness returns a per-sample scalar.
        c, dur = 1, 4.
        sr = 16000
        audio = get_batch_white_noise(1, c, int(sr * dur)).squeeze(0)  # [C, T]
        norm_audio = normalize_audio(audio, strategy='loudness', sample_rate=sr)
        assert norm_audio.isfinite().all()

    def test_normalize_audio_none_strategy(self):
        b, c, dur = 2, 1, 4.
        sr = 3
        # audio already within [-1, 1], 'none' strategy should return it unchanged
        audio = 0.5 * get_batch_white_noise(b, c, int(sr * dur))
        norm_audio = normalize_audio(audio, strategy='none')
        assert torch.allclose(norm_audio, audio)

    def test_normalize_audio_no_normalize_when_below_threshold(self):
        # When normalize=False, audio that is already below peak threshold
        # should be returned without rescaling.
        b, c, dur = 2, 1, 4.
        sr = 3
        audio = 0.1 * get_batch_white_noise(b, c, int(sr * dur))
        norm_audio = normalize_audio(audio, normalize=False, strategy='peak')
        assert torch.allclose(norm_audio, audio)


class TestPcmConversion:

    def test_f32_pcm_int32(self):
        # int32 input should be scaled into float32 [-1, 1]
        audio_i32 = torch.randint(-(2 ** 31), 2 ** 31 - 1, (2, 1, 100), dtype=torch.int32)
        audio_f32 = f32_pcm(audio_i32)
        assert audio_f32.dtype == torch.float32
        assert audio_f32.abs().max() <= 1.0

    def test_f32_pcm_float_passthrough(self):
        audio = torch.rand(2, 1, 100)
        out = f32_pcm(audio)
        assert out is audio  # no copy for float input

    def test_i16_pcm_clipping_prevention(self):
        # values very close to 1.0 should not overflow int16
        audio = torch.ones(1, 1, 10)
        out = i16_pcm(audio)
        assert out.dtype == torch.int16
        assert out.abs().max() <= 2 ** 15 - 1
