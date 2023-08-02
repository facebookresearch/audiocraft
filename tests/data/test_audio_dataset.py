# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from itertools import product
import json
import math
import os
import random
import typing as tp

import pytest
import torch
from torch.utils.data import DataLoader

from audiocraft.data.audio_dataset import (
    AudioDataset,
    AudioMeta,
    _get_audio_meta,
    load_audio_meta,
    save_audio_meta
)
from audiocraft.data.zip import PathInZip

from ..common_utils import TempDirMixin, get_white_noise, save_wav


class TestAudioMeta(TempDirMixin):

    def test_get_audio_meta(self):
        sample_rates = [8000, 16_000]
        channels = [1, 2]
        duration = 1.
        for sample_rate, ch in product(sample_rates, channels):
            n_frames = int(duration * sample_rate)
            wav = get_white_noise(ch, n_frames)
            path = self.get_temp_path('sample.wav')
            save_wav(path, wav, sample_rate)
            m = _get_audio_meta(path, minimal=True)
            assert m.path == path, 'path does not match'
            assert m.sample_rate == sample_rate, 'sample rate does not match'
            assert m.duration == duration, 'duration does not match'
            assert m.amplitude is None
            assert m.info_path is None

    def test_save_audio_meta(self):
        audio_meta = [
            AudioMeta("mypath1", 1., 16_000, None, None, PathInZip('/foo/bar.zip:/relative/file1.json')),
            AudioMeta("mypath2", 2., 16_000, None, None, PathInZip('/foo/bar.zip:/relative/file2.json'))
            ]
        empty_audio_meta = []
        for idx, meta in enumerate([audio_meta, empty_audio_meta]):
            path = self.get_temp_path(f'data_{idx}_save.jsonl')
            save_audio_meta(path, meta)
            with open(path, 'r') as f:
                lines = f.readlines()
                read_meta = [AudioMeta.from_dict(json.loads(line)) for line in lines]
                assert len(read_meta) == len(meta)
                for m, read_m in zip(meta, read_meta):
                    assert m == read_m

    def test_load_audio_meta(self):
        try:
            import dora
        except ImportError:
            dora = None  # type: ignore

        audio_meta = [
            AudioMeta("mypath1", 1., 16_000, None, None, PathInZip('/foo/bar.zip:/relative/file1.json')),
            AudioMeta("mypath2", 2., 16_000, None, None, PathInZip('/foo/bar.zip:/relative/file2.json'))
            ]
        empty_meta = []
        for idx, meta in enumerate([audio_meta, empty_meta]):
            path = self.get_temp_path(f'data_{idx}_load.jsonl')
            with open(path, 'w') as f:
                for m in meta:
                    json_str = json.dumps(m.to_dict()) + '\n'
                    f.write(json_str)
            read_meta = load_audio_meta(path)
            assert len(read_meta) == len(meta)
            for m, read_m in zip(meta, read_meta):
                if dora:
                    m.path = dora.git_save.to_absolute_path(m.path)
                assert m == read_m, f'original={m}, read={read_m}'


class TestAudioDataset(TempDirMixin):

    def _create_audio_files(self,
                            root_name: str,
                            num_examples: int,
                            durations: tp.Union[float, tp.Tuple[float, float]] = (0.1, 1.),
                            sample_rate: int = 16_000,
                            channels: int = 1):
        root_dir = self.get_temp_dir(root_name)
        for i in range(num_examples):
            if isinstance(durations, float):
                duration = durations
            elif isinstance(durations, tuple) and len(durations) == 1:
                duration = durations[0]
            elif isinstance(durations, tuple) and len(durations) == 2:
                duration = random.uniform(durations[0], durations[1])
            else:
                assert False
            n_frames = int(duration * sample_rate)
            wav = get_white_noise(channels, n_frames)
            path = os.path.join(root_dir, f'example_{i}.wav')
            save_wav(path, wav, sample_rate)
        return root_dir

    def _create_audio_dataset(self,
                              root_name: str,
                              total_num_examples: int,
                              durations: tp.Union[float, tp.Tuple[float, float]] = (0.1, 1.),
                              sample_rate: int = 16_000,
                              channels: int = 1,
                              segment_duration: tp.Optional[float] = None,
                              num_examples: int = 10,
                              shuffle: bool = True,
                              return_info: bool = False):
        root_dir = self._create_audio_files(root_name, total_num_examples, durations, sample_rate, channels)
        dataset = AudioDataset.from_path(root_dir,
                                         minimal_meta=True,
                                         segment_duration=segment_duration,
                                         num_samples=num_examples,
                                         sample_rate=sample_rate,
                                         channels=channels,
                                         shuffle=shuffle,
                                         return_info=return_info)
        return dataset

    def test_dataset_full(self):
        total_examples = 10
        min_duration, max_duration = 1., 4.
        sample_rate = 16_000
        channels = 1
        dataset = self._create_audio_dataset(
            'dset', total_examples, durations=(min_duration, max_duration),
            sample_rate=sample_rate, channels=channels, segment_duration=None)
        assert len(dataset) == total_examples
        assert dataset.sample_rate == sample_rate
        assert dataset.channels == channels
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert sample.shape[0] == channels
            assert sample.shape[1] <= int(max_duration * sample_rate)
            assert sample.shape[1] >= int(min_duration * sample_rate)

    def test_dataset_segment(self):
        total_examples = 10
        num_samples = 20
        min_duration, max_duration = 1., 4.
        segment_duration = 1.
        sample_rate = 16_000
        channels = 1
        dataset = self._create_audio_dataset(
            'dset', total_examples, durations=(min_duration, max_duration), sample_rate=sample_rate,
            channels=channels, segment_duration=segment_duration, num_examples=num_samples)
        assert len(dataset) == num_samples
        assert dataset.sample_rate == sample_rate
        assert dataset.channels == channels
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert sample.shape[0] == channels
            assert sample.shape[1] == int(segment_duration * sample_rate)

    def test_dataset_equal_audio_and_segment_durations(self):
        total_examples = 1
        num_samples = 2
        audio_duration = 1.
        segment_duration = 1.
        sample_rate = 16_000
        channels = 1
        dataset = self._create_audio_dataset(
            'dset', total_examples, durations=audio_duration, sample_rate=sample_rate,
            channels=channels, segment_duration=segment_duration, num_examples=num_samples)
        assert len(dataset) == num_samples
        assert dataset.sample_rate == sample_rate
        assert dataset.channels == channels
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert sample.shape[0] == channels
            assert sample.shape[1] == int(segment_duration * sample_rate)
        # the random seek_time adds variability on audio read
        sample_1 = dataset[0]
        sample_2 = dataset[1]
        assert not torch.allclose(sample_1, sample_2)

    def test_dataset_samples(self):
        total_examples = 1
        num_samples = 2
        audio_duration = 1.
        segment_duration = 1.
        sample_rate = 16_000
        channels = 1

        create_dataset = partial(
            self._create_audio_dataset,
            'dset', total_examples, durations=audio_duration, sample_rate=sample_rate,
            channels=channels, segment_duration=segment_duration, num_examples=num_samples,
        )

        dataset = create_dataset(shuffle=True)
        # when shuffle = True, we have different inputs for the same index across epoch
        sample_1 = dataset[0]
        sample_2 = dataset[0]
        assert not torch.allclose(sample_1, sample_2)

        dataset_noshuffle = create_dataset(shuffle=False)
        # when shuffle = False, we have same inputs for the same index across epoch
        sample_1 = dataset_noshuffle[0]
        sample_2 = dataset_noshuffle[0]
        assert torch.allclose(sample_1, sample_2)

    def test_dataset_return_info(self):
        total_examples = 10
        num_samples = 20
        min_duration, max_duration = 1., 4.
        segment_duration = 1.
        sample_rate = 16_000
        channels = 1
        dataset = self._create_audio_dataset(
            'dset', total_examples, durations=(min_duration, max_duration), sample_rate=sample_rate,
            channels=channels, segment_duration=segment_duration, num_examples=num_samples, return_info=True)
        assert len(dataset) == num_samples
        assert dataset.sample_rate == sample_rate
        assert dataset.channels == channels
        for idx in range(len(dataset)):
            sample, segment_info = dataset[idx]
            assert sample.shape[0] == channels
            assert sample.shape[1] == int(segment_duration * sample_rate)
            assert segment_info.sample_rate == sample_rate
            assert segment_info.total_frames == int(segment_duration * sample_rate)
            assert segment_info.n_frames <= int(segment_duration * sample_rate)
            assert segment_info.seek_time >= 0

    def test_dataset_return_info_no_segment_duration(self):
        total_examples = 10
        num_samples = 20
        min_duration, max_duration = 1., 4.
        segment_duration = None
        sample_rate = 16_000
        channels = 1
        dataset = self._create_audio_dataset(
            'dset', total_examples, durations=(min_duration, max_duration), sample_rate=sample_rate,
            channels=channels, segment_duration=segment_duration, num_examples=num_samples, return_info=True)
        assert len(dataset) == total_examples
        assert dataset.sample_rate == sample_rate
        assert dataset.channels == channels
        for idx in range(len(dataset)):
            sample, segment_info = dataset[idx]
            assert sample.shape[0] == channels
            assert sample.shape[1] == segment_info.total_frames
            assert segment_info.sample_rate == sample_rate
            assert segment_info.n_frames <= segment_info.total_frames

    def test_dataset_collate_fn(self):
        total_examples = 10
        num_samples = 20
        min_duration, max_duration = 1., 4.
        segment_duration = 1.
        sample_rate = 16_000
        channels = 1
        dataset = self._create_audio_dataset(
            'dset', total_examples, durations=(min_duration, max_duration), sample_rate=sample_rate,
            channels=channels, segment_duration=segment_duration, num_examples=num_samples, return_info=False)
        batch_size = 4
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0
        )
        for idx, batch in enumerate(dataloader):
            assert batch.shape[0] == batch_size

    @pytest.mark.parametrize("segment_duration", [1.0, None])
    def test_dataset_with_meta_collate_fn(self, segment_duration):
        total_examples = 10
        num_samples = 20
        min_duration, max_duration = 1., 4.
        segment_duration = 1.
        sample_rate = 16_000
        channels = 1
        dataset = self._create_audio_dataset(
            'dset', total_examples, durations=(min_duration, max_duration), sample_rate=sample_rate,
            channels=channels, segment_duration=segment_duration, num_examples=num_samples, return_info=True)
        batch_size = 4
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collater,
            num_workers=0
        )
        for idx, batch in enumerate(dataloader):
            wav, infos = batch
            assert wav.shape[0] == batch_size
            assert len(infos) == batch_size

    @pytest.mark.parametrize("segment_duration,sample_on_weight,sample_on_duration,a_hist,b_hist,c_hist", [
        [1, True, True, 0.5, 0.5, 0.0],
        [1, False, True, 0.25, 0.5, 0.25],
        [1, True, False, 0.666, 0.333, 0.0],
        [1, False, False, 0.333, 0.333, 0.333],
        [None, False, False, 0.333, 0.333, 0.333]])
    def test_sample_with_weight(self, segment_duration, sample_on_weight, sample_on_duration, a_hist, b_hist, c_hist):
        random.seed(1234)
        rng = torch.Generator()
        rng.manual_seed(1234)

        def _get_histogram(dataset, repetitions=20_000):
            counts = {file_meta.path: 0. for file_meta in meta}
            for _ in range(repetitions):
                file_meta = dataset.sample_file(0, rng)
                counts[file_meta.path] += 1
            return {name: count / repetitions for name, count in counts.items()}

        meta = [
           AudioMeta(path='a', duration=5, sample_rate=1, weight=2),
           AudioMeta(path='b', duration=10, sample_rate=1, weight=None),
           AudioMeta(path='c', duration=5, sample_rate=1, weight=0),
        ]
        dataset = AudioDataset(
            meta, segment_duration=segment_duration, sample_on_weight=sample_on_weight,
            sample_on_duration=sample_on_duration)
        hist = _get_histogram(dataset)
        assert math.isclose(hist['a'], a_hist, abs_tol=0.01)
        assert math.isclose(hist['b'], b_hist, abs_tol=0.01)
        assert math.isclose(hist['c'], c_hist, abs_tol=0.01)

    def test_meta_duration_filter_all(self):
        meta = [
           AudioMeta(path='a', duration=5, sample_rate=1, weight=2),
           AudioMeta(path='b', duration=10, sample_rate=1, weight=None),
           AudioMeta(path='c', duration=5, sample_rate=1, weight=0),
        ]
        try:
            AudioDataset(meta, segment_duration=11, min_segment_ratio=1)
            assert False
        except AssertionError:
            assert True

    def test_meta_duration_filter_long(self):
        meta = [
           AudioMeta(path='a', duration=5, sample_rate=1, weight=2),
           AudioMeta(path='b', duration=10, sample_rate=1, weight=None),
           AudioMeta(path='c', duration=5, sample_rate=1, weight=0),
        ]
        dataset = AudioDataset(meta, segment_duration=None, min_segment_ratio=1, max_audio_duration=7)
        assert len(dataset) == 2
