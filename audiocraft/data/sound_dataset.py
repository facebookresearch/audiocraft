# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset of audio with a simple description.
"""

from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
import random
import typing as tp

import numpy as np
import torch

from .info_audio_dataset import (
    InfoAudioDataset,
    get_keyword_or_keyword_list
)
from ..modules.conditioners import (
    ConditioningAttributes,
    SegmentWithAttributes,
    WavCondition,
)


EPS = torch.finfo(torch.float32).eps
TARGET_LEVEL_LOWER = -35
TARGET_LEVEL_UPPER = -15


@dataclass
class SoundInfo(SegmentWithAttributes):
    """Segment info augmented with Sound metadata.
    """
    description: tp.Optional[str] = None
    self_wav: tp.Optional[torch.Tensor] = None

    @property
    def has_sound_meta(self) -> bool:
        return self.description is not None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()

        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == 'self_wav':
                out.wav[key] = value
            else:
                out.text[key] = value
        return out

    @staticmethod
    def attribute_getter(attribute):
        if attribute == 'description':
            preprocess_func = get_keyword_or_keyword_list
        else:
            preprocess_func = None
        return preprocess_func

    @classmethod
    def from_dict(cls, dictionary: dict, fields_required: bool = False):
        _dictionary: tp.Dict[str, tp.Any] = {}

        # allow a subset of attributes to not be loaded from the dictionary
        # these attributes may be populated later
        post_init_attributes = ['self_wav']

        for _field in fields(cls):
            if _field.name in post_init_attributes:
                continue
            elif _field.name not in dictionary:
                if fields_required:
                    raise KeyError(f"Unexpected missing key: {_field.name}")
            else:
                preprocess_func: tp.Optional[tp.Callable] = cls.attribute_getter(_field.name)
                value = dictionary[_field.name]
                if preprocess_func:
                    value = preprocess_func(value)
                _dictionary[_field.name] = value
        return cls(**_dictionary)


class SoundDataset(InfoAudioDataset):
    """Sound audio dataset: Audio dataset with environmental sound-specific metadata.

    Args:
        info_fields_required (bool): Whether all the mandatory metadata fields should be in the loaded metadata.
        external_metadata_source (tp.Optional[str]): Folder containing JSON metadata for the corresponding dataset.
            The metadata files contained in this folder are expected to match the stem of the audio file with
            a json extension.
        aug_p (float): Probability of performing audio mixing augmentation on the batch.
        mix_p (float): Proportion of batch items that are mixed together when applying audio mixing augmentation.
        mix_snr_low (int): Lowerbound for SNR value sampled for mixing augmentation.
        mix_snr_high (int): Upperbound for SNR value sampled for mixing augmentation.
        mix_min_overlap (float): Minimum overlap between audio files when performing mixing augmentation.
        kwargs: Additional arguments for AudioDataset.

    See `audiocraft.data.info_audio_dataset.InfoAudioDataset` for full initialization arguments.
    """
    def __init__(
        self,
        *args,
        info_fields_required: bool = True,
        external_metadata_source: tp.Optional[str] = None,
        aug_p: float = 0.,
        mix_p: float = 0.,
        mix_snr_low: int = -5,
        mix_snr_high: int = 5,
        mix_min_overlap: float = 0.5,
        **kwargs
    ):
        kwargs['return_info'] = True  # We require the info for each song of the dataset.
        super().__init__(*args, **kwargs)
        self.info_fields_required = info_fields_required
        self.external_metadata_source = external_metadata_source
        self.aug_p = aug_p
        self.mix_p = mix_p
        if self.aug_p > 0:
            assert self.mix_p > 0, "Expecting some mixing proportion mix_p if aug_p > 0"
            assert self.channels == 1, "SoundDataset with audio mixing considers only monophonic audio"
        self.mix_snr_low = mix_snr_low
        self.mix_snr_high = mix_snr_high
        self.mix_min_overlap = mix_min_overlap

    def _get_info_path(self, path: tp.Union[str, Path]) -> Path:
        """Get path of JSON with metadata (description, etc.).
        If there exists a JSON with the same name as 'path.name', then it will be used.
        Else, such JSON will be searched for in an external json source folder if it exists.
        """
        info_path = Path(path).with_suffix('.json')
        if Path(info_path).exists():
            return info_path
        elif self.external_metadata_source and (Path(self.external_metadata_source) / info_path.name).exists():
            return Path(self.external_metadata_source) / info_path.name
        else:
            raise Exception(f"Unable to find a metadata JSON for path: {path}")

    def __getitem__(self, index):
        wav, info = super().__getitem__(index)
        info_data = info.to_dict()
        info_path = self._get_info_path(info.meta.path)
        if Path(info_path).exists():
            with open(info_path, 'r') as json_file:
                sound_data = json.load(json_file)
                sound_data.update(info_data)
                sound_info = SoundInfo.from_dict(sound_data, fields_required=self.info_fields_required)
                # if there are multiple descriptions, sample one randomly
                if isinstance(sound_info.description, list):
                    sound_info.description = random.choice(sound_info.description)
        else:
            sound_info = SoundInfo.from_dict(info_data, fields_required=False)

        sound_info.self_wav = WavCondition(
            wav=wav[None], length=torch.tensor([info.n_frames]),
            sample_rate=[sound_info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])

        return wav, sound_info

    def collater(self, samples):
        # when training, audio mixing is performed in the collate function
        wav, sound_info = super().collater(samples)  # SoundDataset always returns infos
        if self.aug_p > 0:
            wav, sound_info = mix_samples(wav, sound_info, self.aug_p, self.mix_p,
                                          snr_low=self.mix_snr_low, snr_high=self.mix_snr_high,
                                          min_overlap=self.mix_min_overlap)
        return wav, sound_info


def rms_f(x: torch.Tensor) -> torch.Tensor:
    return (x ** 2).mean(1).pow(0.5)


def normalize(audio: torch.Tensor, target_level: int = -25) -> torch.Tensor:
    """Normalize the signal to the target level."""
    rms = rms_f(audio)
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    audio = audio * scalar.unsqueeze(1)
    return audio


def is_clipped(audio: torch.Tensor, clipping_threshold: float = 0.99) -> torch.Tensor:
    return (abs(audio) > clipping_threshold).any(1)


def mix_pair(src: torch.Tensor, dst: torch.Tensor, min_overlap: float) -> torch.Tensor:
    start = random.randint(0, int(src.shape[1] * (1 - min_overlap)))
    remainder = src.shape[1] - start
    if dst.shape[1] > remainder:
        src[:, start:] = src[:, start:] + dst[:, :remainder]
    else:
        src[:, start:start+dst.shape[1]] = src[:, start:start+dst.shape[1]] + dst
    return src


def snr_mixer(clean: torch.Tensor, noise: torch.Tensor, snr: int, min_overlap: float,
              target_level: int = -25, clipping_threshold: float = 0.99) -> torch.Tensor:
    """Function to mix clean speech and noise at various SNR levels.

    Args:
        clean (torch.Tensor): Clean audio source to mix, of shape [B, T].
        noise (torch.Tensor): Noise audio source to mix, of shape [B, T].
        snr (int): SNR level when mixing.
        min_overlap (float): Minimum overlap between the two mixed sources.
        target_level (int): Gain level in dB.
        clipping_threshold (float): Threshold for clipping the audio.
    Returns:
        torch.Tensor: The mixed audio, of shape [B, T].
    """
    if clean.shape[1] > noise.shape[1]:
        noise = torch.nn.functional.pad(noise, (0, clean.shape[1] - noise.shape[1]))
    else:
        noise = noise[:, :clean.shape[1]]

    # normalizing to -25 dB FS
    clean = clean / (clean.max(1)[0].abs().unsqueeze(1) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = rms_f(clean)

    noise = noise / (noise.max(1)[0].abs().unsqueeze(1) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = rms_f(noise)

    # set the noise level for a given SNR
    noisescalar = (rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)).unsqueeze(1)
    noisenewlevel = noise * noisescalar

    # mix noise and clean speech
    noisyspeech = mix_pair(clean, noisenewlevel, min_overlap)

    # randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # there is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(TARGET_LEVEL_LOWER, TARGET_LEVEL_UPPER)
    rmsnoisy = rms_f(noisyspeech)
    scalarnoisy = (10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)).unsqueeze(1)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    clipped = is_clipped(noisyspeech)
    if clipped.any():
        noisyspeech_maxamplevel = noisyspeech[clipped].max(1)[0].abs().unsqueeze(1) / (clipping_threshold - EPS)
        noisyspeech[clipped] = noisyspeech[clipped] / noisyspeech_maxamplevel

    return noisyspeech


def snr_mix(src: torch.Tensor, dst: torch.Tensor, snr_low: int, snr_high: int, min_overlap: float):
    if snr_low == snr_high:
        snr = snr_low
    else:
        snr = np.random.randint(snr_low, snr_high)
    mix = snr_mixer(src, dst, snr, min_overlap)
    return mix


def mix_text(src_text: str, dst_text: str):
    """Mix text from different sources by concatenating them."""
    if src_text == dst_text:
        return src_text
    return src_text + " " + dst_text


def mix_samples(wavs: torch.Tensor, infos: tp.List[SoundInfo], aug_p: float, mix_p: float,
                snr_low: int, snr_high: int, min_overlap: float):
    """Mix samples within a batch, summing the waveforms and concatenating the text infos.

    Args:
        wavs (torch.Tensor): Audio tensors of shape [B, C, T].
        infos (list[SoundInfo]): List of SoundInfo items corresponding to the audio.
        aug_p (float): Augmentation probability.
        mix_p (float): Proportion of items in the batch to mix (and merge) together.
        snr_low (int): Lowerbound for sampling SNR.
        snr_high (int): Upperbound for sampling SNR.
        min_overlap (float): Minimum overlap between mixed samples.
    Returns:
        tuple[torch.Tensor, list[SoundInfo]]: A tuple containing the mixed wavs
            and mixed SoundInfo for the given batch.
    """
    # no mixing to perform within the batch
    if mix_p == 0:
        return wavs, infos

    if random.uniform(0, 1) < aug_p:
        # perform all augmentations on waveforms as [B, T]
        # randomly picking pairs of audio to mix
        assert wavs.size(1) == 1, f"Mix samples requires monophonic audio but C={wavs.size(1)}"
        wavs = wavs.mean(dim=1, keepdim=False)
        B, T = wavs.shape
        k = int(mix_p * B)
        mixed_sources_idx = torch.randperm(B)[:k]
        mixed_targets_idx = torch.randperm(B)[:k]
        aug_wavs = snr_mix(
            wavs[mixed_sources_idx],
            wavs[mixed_targets_idx],
            snr_low,
            snr_high,
            min_overlap,
        )
        # mixing textual descriptions in metadata
        descriptions = [info.description for info in infos]
        aug_infos = []
        for i, j in zip(mixed_sources_idx, mixed_targets_idx):
            text = mix_text(descriptions[i], descriptions[j])
            m = replace(infos[i])
            m.description = text
            aug_infos.append(m)

        # back to [B, C, T]
        aug_wavs = aug_wavs.unsqueeze(1)
        assert aug_wavs.shape[0] > 0, "Samples mixing returned empty batch."
        assert aug_wavs.dim() == 3, f"Returned wav should be [B, C, T] but dim = {aug_wavs.dim()}"
        assert aug_wavs.shape[0] == len(aug_infos), "Mismatch between number of wavs and infos in the batch"

        return aug_wavs, aug_infos  # [B, C, T]
    else:
        # randomly pick samples in the batch to match
        # the batch size when performing audio mixing
        B, C, T = wavs.shape
        k = int(mix_p * B)
        wav_idx = torch.randperm(B)[:k]
        wavs = wavs[wav_idx]
        infos = [infos[i] for i in wav_idx]
        assert wavs.shape[0] == len(infos), "Mismatch between number of wavs and infos in the batch"

        return wavs, infos  # [B, C, T]
