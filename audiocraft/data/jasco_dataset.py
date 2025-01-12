# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import bisect
import pickle
import math
import os
import torch
import typing as tp
from pathlib import Path
from dataclasses import dataclass, fields
from ..utils.utils import construct_frame_chords
from .music_dataset import MusicDataset, MusicInfo
from .audio_dataset import load_audio_meta
from ..modules.conditioners import (ConditioningAttributes, SymbolicCondition)
import librosa
import numpy as np


@dataclass
class JascoInfo(MusicInfo):
    """
    A data class extending MusicInfo for JASCO. The following attributes are added:
    Attributes:
        frame_chords (Optional[list]): A list of chords associated with frames in the music piece.
    """
    chords: tp.Optional[SymbolicCondition] = None
    melody: tp.Optional[SymbolicCondition] = None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()
        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == 'self_wav':
                out.wav[key] = value
            elif key in {'chords', 'melody'}:
                out.symbolic[key] = value
            elif key == 'joint_embed':
                for embed_attribute, embed_cond in value.items():
                    out.joint_embed[embed_attribute] = embed_cond
            else:
                if isinstance(value, list):
                    value = ' '.join(value)
                out.text[key] = value
        return out


class MelodyData:

    SALIENCE_MODEL_EXPECTED_SAMPLE_RATE = 22050
    SALIENCE_MODEL_EXPECTED_HOP_SIZE = 256

    def __init__(self,
                 latent_fr: int,
                 segment_duration: float,
                 melody_fr: int = 86,
                 melody_salience_dim: int = 53,
                 chroma_root: tp.Optional[str] = None,
                 override_cache: bool = False,
                 do_argmax: bool = True):
        """Module to load salience matrix for a given info.

        Args:
            latent_fr (int): latent frame rate to match (interpolates model frame rate accordingly).
            segment_duration (float): expected segment duration.
            melody_fr (int, optional): extracted salience frame rate. Defaults to 86.
            melody_salience_dim (int, optional): salience dim. Defaults to 53.
            chroma_root (str, optional): path to root containing salience cache. Defaults to None.
            override_cache (bool, optional): rewrite cache. Defaults to False.
            do_argmax (bool, optional): argmax the melody matrix. Defaults to True.
        """

        self.segment_duration = segment_duration
        self.melody_fr = melody_fr
        self.latent_fr = latent_fr
        self.melody_salience_dim = melody_salience_dim
        self.do_argmax = do_argmax
        self.tgt_chunk_len = int(latent_fr * segment_duration)

        self.null_op = False
        if chroma_root is None:
            self.null_op = True
        elif not os.path.exists(f"{chroma_root}/cache.pkl") or override_cache:
            self.tracks = []
            for file in librosa.util.find_files(chroma_root, ext='txt'):
                with open(file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        self.tracks.append(line.strip())

            # go over tracks and add the corresponding saliency file to self.saliency_files
            self.saliency_files = []
            for track in self.tracks:
                # saliency file name
                salience_file = f"{chroma_root}/{track.split('/')[-1].split('.')[0]}_multif0_salience.npz"
                assert os.path.exists(salience_file), f"File {salience_file} does not exist"
                self.saliency_files.append(salience_file)

            self.trk2idx = {trk.split('/')[-1].split('.')[0]: i for i, trk in enumerate(self.tracks)}
            torch.save({'tracks': self.tracks,
                        'saliency_files': self.saliency_files,
                        'trk2idx': self.trk2idx}, f"{chroma_root}/cache.pkl")
        else:
            tmp = torch.load(f"{chroma_root}/cache.pkl")
            self.tracks = tmp['tracks']
            self.saliency_files = tmp['saliency_files']
            self.trk2idx = tmp['trk2idx']
        self.model_frame_rate = int(self.SALIENCE_MODEL_EXPECTED_SAMPLE_RATE / self.SALIENCE_MODEL_EXPECTED_HOP_SIZE)

    def load_saliency_from_saliency_dict(self,
                                         saliency_dict: tp.Dict[str, tp.Any],
                                         offset: float) -> torch.Tensor:
        """
        construct the salience matrix and perform linear interpolation w.r.t the temporal axis to match the expected
        frame rate.
        """
        # get saliency map for the segment
        saliency_dict_ = {}
        l, r = int(offset * self.model_frame_rate), int((offset + self.segment_duration) * self.model_frame_rate)
        saliency_dict_['salience'] = saliency_dict['salience'][:, l: r].T
        saliency_dict_['times'] = saliency_dict['times'][l: r] - offset
        saliency_dict_['freqs'] = saliency_dict['freqs']

        saliency_dict_['salience'] = torch.Tensor(saliency_dict_['salience']).float().permute(1, 0)  # C, T
        if saliency_dict_['salience'].shape[-1] <= int(self.model_frame_rate) / self.latent_fr:  # empty chroma
            saliency_dict_['salience'] = torch.zeros((saliency_dict_['salience'].shape[0], self.tgt_chunk_len))
        else:
            salience = torch.nn.functional.interpolate(saliency_dict_['salience'].unsqueeze(0),
                                                       scale_factor=self.latent_fr/int(self.model_frame_rate),
                                                       mode='linear').squeeze(0)
            if salience.shape[-1] < self.tgt_chunk_len:
                salience = torch.nn.functional.pad(salience,
                                                   (0, self.tgt_chunk_len - salience.shape[-1]),
                                                   mode='constant',
                                                   value=0)
            elif salience.shape[-1] > self.tgt_chunk_len:
                salience = salience[..., :self.tgt_chunk_len]
            saliency_dict_['salience'] = salience

        salience = saliency_dict_['salience']
        if self.do_argmax:
            binary_mask = torch.zeros_like(salience)
            binary_mask[torch.argmax(salience, dim=0), torch.arange(salience.shape[-1])] = 1
            binary_mask *= (salience != 0).float()
            salience = binary_mask
        return salience

    def get_null_salience(self) -> torch.Tensor:
        return torch.zeros((self.melody_salience_dim, self.tgt_chunk_len))

    def __call__(self, x: MusicInfo) -> torch.Tensor:
        """Reads salience matrix from memory, shifted by seek time

        Args:
            x (MusicInfo): Music info of a single sample

        Returns:
            torch.Tensor: salience matrix matching the target info
        """
        fname: str = x.meta.path.split("/")[-1].split(".")[0] if x.meta.path is not None else ""
        if x.meta.path is None or x.meta.path == "" or fname not in self.trk2idx:
            salience = self.get_null_salience()
        else:
            assert fname in self.trk2idx, f"Track {fname} not found in the cache"
            idx = self.trk2idx[fname]
            saliency_dict = np.load(self.saliency_files[idx], allow_pickle=True)
            salience = self.load_saliency_from_saliency_dict(saliency_dict, x.seek_time)
        return salience


class JascoDataset(MusicDataset):
    """JASCO dataset is a MusicDataset with jasco-related symbolic data (chords, melody).

    Args:
        chords_card (int): The cardinality of the chords, default is 194.
        compression_model_framerate (int): The framerate for the compression model, default is 50.

    See `audiocraft.data.info_audio_dataset.MusicDataset` for full initialization arguments.
    """
    @classmethod
    def from_meta(cls, root: tp.Union[str, Path], **kwargs):
        """Instantiate AudioDataset from a path to a directory containing a manifest as a jsonl file.

        Args:
            root (str or Path): Path to root folder containing audio files.
            kwargs: Additional keyword arguments for the AudioDataset.
        """
        root = Path(root)
        # a directory is given
        if root.is_dir():
            if (root / 'data.jsonl').exists():
                meta_json = root / 'data.jsonl'
            elif (root / 'data.jsonl.gz').exists():
                meta_json = root / 'data.jsonl.gz'
            else:
                raise ValueError("Don't know where to read metadata from in the dir. "
                                 "Expecting either a data.jsonl or data.jsonl.gz file but none found.")
        # jsonl file was specified
        else:
            assert root.exists() and root.suffix == '.jsonl', \
                "Either specified path not exist or it is not a jsonl format"
            meta_json = root
            root = root.parent
        meta = load_audio_meta(meta_json)
        kwargs['root'] = root
        return cls(meta, **kwargs)

    def __init__(self, *args,
                 chords_card: int = 194,
                 compression_model_framerate: float = 50.,
                 melody_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = {},
                 **kwargs):
        """Dataset class for text-to-music generation with temporal controls as in
        (JASCO)[https://arxiv.org/pdf/2406.10970]

        Args:
            chords_card (int, optional): Number of chord ebeddings. Defaults to 194.
            compression_model_framerate (float, optional): Expected frame rate of the resulted latent. Defaults to 50.
            melody_kwargs (tp.Optional[tp.Dict[str, tp.Any]], optional): See MelodyData class. Defaults to {}.
        """
        root = kwargs.pop('root')
        super().__init__(*args, **kwargs)

        chords_mapping_path = root / 'chord_to_index_mapping.pkl'
        chords_path = root / 'chords_per_track.pkl'
        self.mapping_dict = pickle.load(open(chords_mapping_path, "rb")) if \
            os.path.exists(chords_mapping_path) else None

        self.chords_per_track = pickle.load(open(chords_path, "rb")) if \
            os.path.exists(chords_path) else None

        self.compression_model_framerate = compression_model_framerate
        self.null_chord_idx = chords_card

        self.melody_module = MelodyData(**melody_kwargs)  # type: ignore

    def _get_relevant_sublist(self, chords, timestamp):
        """
        Returns the sublist of chords within the specified timestamp and segment length.

        Args:
            chords (list): A sorted list of tuples containing (time changed, chord).
            timestamp (float): The timestamp at which to start the sublist.

        Returns:
            list: A list of chords within the specified timestamp and segment length.
        """
        end_time = timestamp + self.segment_duration

        # Use binary search to find the starting index of the relevant sublist
        start_index = bisect.bisect_left(chords, (timestamp,))

        if start_index != 0:
            prev_chord = chords[start_index - 1]
        else:
            prev_chord = (0.0, "N")

        relevant_chords = []

        for time_changed, chord in chords[start_index:]:
            if time_changed >= end_time:
                break
            relevant_chords.append((time_changed, chord))

        return relevant_chords, prev_chord

    def _get_chords(self, music_info: MusicInfo, effective_segment_dur: float) -> torch.Tensor:
        if self.chords_per_track is None:
            # use null chord when there's no chords in dataset
            seq_len = math.ceil(self.compression_model_framerate * effective_segment_dur)
            return torch.ones(seq_len, dtype=int) * self.null_chord_idx  # type: ignore

        fr = self.compression_model_framerate

        idx = music_info.meta.path.split("/")[-1].split(".")[0]
        chords = self.chords_per_track[idx]

        min_timestamp = music_info.seek_time

        chords = [(item[1], item[0]) for item in chords]
        chords, prev_chord = self._get_relevant_sublist(
            chords, min_timestamp
        )

        iter_min_timestamp = int(min_timestamp * fr) + 1

        frame_chords = construct_frame_chords(
            iter_min_timestamp, chords, self.mapping_dict, prev_chord[1],  # type: ignore
            fr, self.segment_duration  # type: ignore
        )

        return torch.tensor(frame_chords)

    def __getitem__(self, index):
        wav, music_info = super().__getitem__(index)
        assert not wav.isinfinite().any(), f"inf detected in wav file: {music_info}"
        wav = wav.float()

        # downcast music info to jasco info
        jasco_info = JascoInfo({k: v for k, v in music_info.__dict__.items()})

        # get chords
        effective_segment_dur = (wav.shape[-1] / self.sample_rate) if \
            self.segment_duration is None else self.segment_duration
        frame_chords = self._get_chords(music_info, effective_segment_dur)
        jasco_info.chords = SymbolicCondition(frame_chords=frame_chords)

        # get melody
        jasco_info.melody = SymbolicCondition(melody=self.melody_module(music_info))
        return wav, jasco_info
