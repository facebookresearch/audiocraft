# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product
import random

import numpy as np
import torch
import torchaudio

from audiocraft.data.audio import audio_info, audio_read, audio_write, _av_read

from ..common_utils import TempDirMixin, get_white_noise, save_wav


class TestInfo(TempDirMixin):

    def test_info_mp3(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        for sample_rate, ch in product(sample_rates, channels):
            wav = get_white_noise(ch, int(sample_rate * duration))
            path = self.get_temp_path('sample_wav.mp3')
            save_wav(path, wav, sample_rate)
            info = audio_info(path)
            assert info.sample_rate == sample_rate
            assert info.channels == ch
            # we cannot trust torchaudio for num_frames, so we don't check

    def _test_info_format(self, ext: str):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(sample_rate * duration)
            wav = get_white_noise(ch, n_frames)
            path = self.get_temp_path(f'sample_wav{ext}')
            save_wav(path, wav, sample_rate)
            info = audio_info(path)
            assert info.sample_rate == sample_rate
            assert info.channels == ch
            assert np.isclose(info.duration, duration, atol=1e-5)

    def test_info_wav(self):
        self._test_info_format('.wav')

    def test_info_flac(self):
        self._test_info_format('.flac')

    def test_info_ogg(self):
        self._test_info_format('.ogg')

    def test_info_m4a(self):
        # TODO: generate m4a file programmatically
        # self._test_info_format('.m4a')
        pass


class TestRead(TempDirMixin):

    def test_read_full_wav(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(sample_rate * duration)
            wav = get_white_noise(ch, n_frames).clamp(-0.99, 0.99)
            path = self.get_temp_path('sample_wav.wav')
            save_wav(path, wav, sample_rate)
            read_wav, read_sr = audio_read(path)
            assert read_sr == sample_rate
            assert read_wav.shape[0] == wav.shape[0]
            assert read_wav.shape[1] == wav.shape[1]
            assert torch.allclose(read_wav, wav, rtol=1e-03, atol=1e-04)

    def test_read_partial_wav(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        read_duration = torch.rand(1).item()
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(sample_rate * duration)
            read_frames = int(sample_rate * read_duration)
            wav = get_white_noise(ch, n_frames).clamp(-0.99, 0.99)
            path = self.get_temp_path('sample_wav.wav')
            save_wav(path, wav, sample_rate)
            read_wav, read_sr = audio_read(path, 0, read_duration)
            assert read_sr == sample_rate
            assert read_wav.shape[0] == wav.shape[0]
            assert read_wav.shape[1] == read_frames
            assert torch.allclose(read_wav[..., 0:read_frames], wav[..., 0:read_frames], rtol=1e-03, atol=1e-04)

    def test_read_seek_time_wav(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        read_duration = 1.
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(sample_rate * duration)
            wav = get_white_noise(ch, n_frames).clamp(-0.99, 0.99)
            path = self.get_temp_path('sample_wav.wav')
            save_wav(path, wav, sample_rate)
            seek_time = torch.rand(1).item()
            read_wav, read_sr = audio_read(path, seek_time, read_duration)
            seek_frames = int(sample_rate * seek_time)
            expected_frames = n_frames - seek_frames
            assert read_sr == sample_rate
            assert read_wav.shape[0] == wav.shape[0]
            assert read_wav.shape[1] == expected_frames
            assert torch.allclose(read_wav, wav[..., seek_frames:], rtol=1e-03, atol=1e-04)

    def test_read_seek_time_wav_padded(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        read_duration = 1.
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(sample_rate * duration)
            read_frames = int(sample_rate * read_duration)
            wav = get_white_noise(ch, n_frames).clamp(-0.99, 0.99)
            path = self.get_temp_path('sample_wav.wav')
            save_wav(path, wav, sample_rate)
            seek_time = torch.rand(1).item()
            seek_frames = int(sample_rate * seek_time)
            expected_frames = n_frames - seek_frames
            read_wav, read_sr = audio_read(path, seek_time, read_duration, pad=True)
            expected_pad_wav = torch.zeros(wav.shape[0], read_frames - expected_frames)
            assert read_sr == sample_rate
            assert read_wav.shape[0] == wav.shape[0]
            assert read_wav.shape[1] == read_frames
            assert torch.allclose(read_wav[..., :expected_frames], wav[..., seek_frames:], rtol=1e-03, atol=1e-04)
            assert torch.allclose(read_wav[..., expected_frames:], expected_pad_wav)


class TestAvRead(TempDirMixin):

    def test_avread_seek_base(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 2.
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(sample_rate * duration)
            wav = get_white_noise(ch, n_frames)
            path = self.get_temp_path(f'reference_a_{sample_rate}_{ch}.wav')
            save_wav(path, wav, sample_rate)
            for _ in range(100):
                # seek will always load a full duration segment in the file
                seek_time = random.uniform(0.0, 1.0)
                seek_duration = random.uniform(0.001, 1.0)
                read_wav, read_sr = _av_read(path, seek_time, seek_duration)
                assert read_sr == sample_rate
                assert read_wav.shape[0] == wav.shape[0]
                assert read_wav.shape[-1] == int(seek_duration * sample_rate)

    def test_avread_seek_partial(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(sample_rate * duration)
            wav = get_white_noise(ch, n_frames)
            path = self.get_temp_path(f'reference_b_{sample_rate}_{ch}.wav')
            save_wav(path, wav, sample_rate)
            for _ in range(100):
                # seek will always load a partial segment
                seek_time = random.uniform(0.5, 1.)
                seek_duration = 1.
                expected_num_frames = n_frames - int(seek_time * sample_rate)
                read_wav, read_sr = _av_read(path, seek_time, seek_duration)
                assert read_sr == sample_rate
                assert read_wav.shape[0] == wav.shape[0]
                assert read_wav.shape[-1] == expected_num_frames

    def test_avread_seek_outofbound(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(sample_rate * duration)
            wav = get_white_noise(ch, n_frames)
            path = self.get_temp_path(f'reference_c_{sample_rate}_{ch}.wav')
            save_wav(path, wav, sample_rate)
            seek_time = 1.5
            read_wav, read_sr = _av_read(path, seek_time, 1.)
            assert read_sr == sample_rate
            assert read_wav.shape[0] == wav.shape[0]
            assert read_wav.shape[-1] == 0

    def test_avread_seek_edge(self):
        sample_rates = [8000, 16_000]
        # some of these values will have
        # int(((frames - 1) / sample_rate) * sample_rate) != (frames - 1)
        n_frames = [1000, 1001, 1002]
        channels = [1, 2]
        for sample_rate, ch, frames in product(sample_rates, channels, n_frames):
            duration = frames / sample_rate
            wav = get_white_noise(ch, frames)
            path = self.get_temp_path(f'reference_d_{sample_rate}_{ch}.wav')
            save_wav(path, wav, sample_rate)
            seek_time = (frames - 1) / sample_rate
            seek_frames = int(seek_time * sample_rate)
            read_wav, read_sr = _av_read(path, seek_time, duration)
            assert read_sr == sample_rate
            assert read_wav.shape[0] == wav.shape[0]
            assert read_wav.shape[-1] == (frames - seek_frames)


class TestAudioWrite(TempDirMixin):

    def test_audio_write_wav(self):
        torch.manual_seed(1234)
        sample_rates = [8000, 16_000]
        n_frames = [1000, 1001, 1002]
        channels = [1, 2]
        strategies = ["peak", "clip", "rms"]
        formats = ["wav", "mp3"]
        for sample_rate, ch, frames in product(sample_rates, channels, n_frames):
            for format_, strategy in product(formats, strategies):
                wav = get_white_noise(ch, frames)
                path = self.get_temp_path(f'pred_{sample_rate}_{ch}')
                audio_write(path, wav, sample_rate, format_, strategy=strategy)
                read_wav, read_sr = torchaudio.load(f'{path}.{format_}')
                if format_ == "wav":
                    assert read_wav.shape == wav.shape

                if format_ == "wav" and strategy in ["peak", "rms"]:
                    rescaled_read_wav = read_wav / read_wav.abs().max() * wav.abs().max()
                    # for a Gaussian, the typical max scale will be less than ~5x the std.
                    # The error when writing to disk will ~ 1/2**15, and when rescaling, 5x that.
                    # For RMS target, rescaling leaves more headroom by default, leading
                    # to a 20x rescaling typically
                    atol = (5 if strategy == "peak" else 20) / 2**15
                    delta = (rescaled_read_wav - wav).abs().max()
                    assert torch.allclose(wav, rescaled_read_wav, rtol=0, atol=atol), (delta, atol)
            formats = ["wav"]  # faster unit tests
