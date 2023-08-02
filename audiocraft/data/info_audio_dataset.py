# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Base classes for the datasets that also provide non-audio metadata,
e.g. description, text transcription etc.
"""
from dataclasses import dataclass
import logging
import math
import re
import typing as tp

import torch

from .audio_dataset import AudioDataset, AudioMeta
from ..environment import AudioCraftEnvironment
from ..modules.conditioners import SegmentWithAttributes, ConditioningAttributes


logger = logging.getLogger(__name__)


def _clusterify_meta(meta: AudioMeta) -> AudioMeta:
    """Monkey-patch meta to match cluster specificities."""
    meta.path = AudioCraftEnvironment.apply_dataset_mappers(meta.path)
    if meta.info_path is not None:
        meta.info_path.zip_path = AudioCraftEnvironment.apply_dataset_mappers(meta.info_path.zip_path)
    return meta


def clusterify_all_meta(meta: tp.List[AudioMeta]) -> tp.List[AudioMeta]:
    """Monkey-patch all meta to match cluster specificities."""
    return [_clusterify_meta(m) for m in meta]


@dataclass
class AudioInfo(SegmentWithAttributes):
    """Dummy SegmentInfo with empty attributes.

    The InfoAudioDataset is expected to return metadata that inherits
    from SegmentWithAttributes class and can return conditioning attributes.

    This basically guarantees all datasets will be compatible with current
    solver that contain conditioners requiring this.
    """
    audio_tokens: tp.Optional[torch.Tensor] = None  # populated when using cached batch for training a LM.

    def to_condition_attributes(self) -> ConditioningAttributes:
        return ConditioningAttributes()


class InfoAudioDataset(AudioDataset):
    """AudioDataset that always returns metadata as SegmentWithAttributes along with the audio waveform.

    See `audiocraft.data.audio_dataset.AudioDataset` for initialization arguments.
    """
    def __init__(self, meta: tp.List[AudioMeta], **kwargs):
        super().__init__(clusterify_all_meta(meta), **kwargs)

    def __getitem__(self, index: int) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, SegmentWithAttributes]]:
        if not self.return_info:
            wav = super().__getitem__(index)
            assert isinstance(wav, torch.Tensor)
            return wav
        wav, meta = super().__getitem__(index)
        return wav, AudioInfo(**meta.to_dict())


def get_keyword_or_keyword_list(value: tp.Optional[str]) -> tp.Union[tp.Optional[str], tp.Optional[tp.List[str]]]:
    """Preprocess a single keyword or possible a list of keywords."""
    if isinstance(value, list):
        return get_keyword_list(value)
    else:
        return get_keyword(value)


def get_string(value: tp.Optional[str]) -> tp.Optional[str]:
    """Preprocess a single keyword."""
    if value is None or (not isinstance(value, str)) or len(value) == 0 or value == 'None':
        return None
    else:
        return value.strip()


def get_keyword(value: tp.Optional[str]) -> tp.Optional[str]:
    """Preprocess a single keyword."""
    if value is None or (not isinstance(value, str)) or len(value) == 0 or value == 'None':
        return None
    else:
        return value.strip().lower()


def get_keyword_list(values: tp.Union[str, tp.List[str]]) -> tp.Optional[tp.List[str]]:
    """Preprocess a list of keywords."""
    if isinstance(values, str):
        values = [v.strip() for v in re.split(r'[,\s]', values)]
    elif isinstance(values, float) and math.isnan(values):
        values = []
    if not isinstance(values, list):
        logger.debug(f"Unexpected keyword list {values}")
        values = [str(values)]

    kws = [get_keyword(v) for v in values]
    kw_list = [k for k in kws if k is not None]
    if len(kw_list) == 0:
        return None
    else:
        return kw_list
