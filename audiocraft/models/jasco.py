# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using JASCO. This will combine all the required components
and provide easy access to the generation API.
"""
import os
import math
import pickle
import torch
import typing as tp

from audiocraft.utils.utils import construct_frame_chords
from .genmodel import BaseGenModel
from .loaders import load_compression_model, load_jasco_model
from ..data.audio_utils import convert_audio
from ..modules.conditioners import WavCondition, ConditioningAttributes, SymbolicCondition, JascoCondConst


class JASCO(BaseGenModel):
    """JASCO main model with convenient generation API.
    Args:
       chords_mapping_path: path to chords to index mapping pickle
       kwargs - See MusicGen class.
    """
    def __init__(self, chords_mapping_path='assets/chord_to_index_mapping.pkl', **kwargs):
        super().__init__(**kwargs)
        # JASCO operates over a fixed sequence length defined in it's config.
        self.duration = self.lm.cfg.dataset.segment_duration

        # load chord2index mapping of Chordino (https://github.com/ohollo/chord-extractor)
        assert os.path.exists(chords_mapping_path)
        self.chords_mapping = pickle.load(open(chords_mapping_path, "rb"))

        # set generation parameters
        self.set_generation_params()

    @staticmethod
    def get_pretrained(name: str = 'facebook/jasco-chords-drums-400M', device=None,
                       chords_mapping_path='assets/chord_to_index_mapping.pkl'):
        """Return pretrained model, we provide 2 models:
        1. facebook/jasco-chords-drums-400M: 10s music generation conditioned on
                                             text, chords and drums, 400M parameters.
        2. facebook/jasco-chords-drums-1B: 10s music generation conditioned on
                                           text, chords and drums, 1B parameters.
        """
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        compression_model = load_compression_model(name, device=device)
        lm = load_jasco_model(name, compression_model, device=device)

        kwargs = {'name': name,
                  'compression_model': compression_model,
                  'lm': lm,
                  'chords_mapping_path': chords_mapping_path}
        return JASCO(**kwargs)

    def set_generation_params(self,
                              cfg_coef_all: float = 5.0,
                              cfg_coef_txt: float = 0.0,
                              **kwargs):
        """Set the generation parameters for JASCO.

        Args:
            cfg_coef_all (float, optional): Coefficient used in multi-source classifier free guidance -
                                            all conditions term. Defaults to 5.0.
            cfg_coef_txt (float, optional): Coefficient used in multi-source classifier free guidance -
                                            text condition term. Defaults to 0.0.

        """
        self.generation_params = {
            'cfg_coef_all': cfg_coef_all,
            'cfg_coef_txt': cfg_coef_txt
        }
        self.generation_params.update(kwargs)

    def _unnormalized_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Unnormalize latents, shifting back to EnCodec's expected mean, std"""
        assert self.cfg is not None
        scaled = latents * self.cfg.compression_model_latent_std
        return scaled + self.cfg.compression_model_latent_mean

    def generate_audio(self, gen_latents: torch.Tensor) -> torch.Tensor:
        """Decode audio from generated latents"""
        assert gen_latents.dim() == 3  # [B, T, C]

        # unnormalize latents
        gen_latents = self._unnormalized_latents(gen_latents)
        return self.compression_model.model.decoder(gen_latents.permute(0, 2, 1))

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate continuous audio latents given conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (here text).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated latents, of shape [B, T, C].
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)

        def _progress_callback(ode_steps: int, max_ode_steps: int):
            ode_steps += 1
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(ode_steps, max_ode_steps)
            else:
                print(f'{ode_steps: 6d} / {max_ode_steps: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        # generate by sampling from the LM
        with self.autocast:
            total_gen_len = math.ceil(self.duration * self.compression_model.frame_rate)
            return self.lm.generate(
                   prompt_tokens, attributes,
                   callback=callback, max_gen_len=total_gen_len, **self.generation_params)

    def _prepare_chord_conditions(
            self,
            attributes: tp.List[ConditioningAttributes],
            chords: tp.Optional[tp.List[tp.Tuple[str, float]]],
    ) -> tp.List[ConditioningAttributes]:
        """
        Prepares chord conditions by translating symbolic chord progressions into a sequence of integers.
        This method updates the ConditioningAttributes with per-frame chords information.
        Args:
            attributes (List[ConditioningAttributes]):
                The initial attributes and optional tensor data.
            chords (List[Tuple[str, float]]):
                A list of tuples containing chord labels and their start times.
        Returns:
            List[ConditioningAttributes]:
                The updated attributes with frame chords integrated, alongside the original optional tensor data.
        """
        if chords is None or chords == []:
            for att in attributes:
                att.symbolic[JascoCondConst.CRD.value] = SymbolicCondition(frame_chords=-1 *
                                                                           torch.ones(1, dtype=torch.int32))
            return attributes

        # flip from (chord, start_time) to (start_time, chord)
        chords_time_first: tp.List[tuple[float, str]] = [(item[1], item[0]) for item in chords]

        # translate symbolic chord progression into a sequence of ints
        frame_chords = construct_frame_chords(min_timestamp=0,
                                              chord_changes=chords_time_first,
                                              mapping_dict=self.chords_mapping,
                                              prev_chord='',
                                              frame_rate=self.compression_model.frame_rate,
                                              segment_duration=self.duration)
        # update the attribute objects
        for att in attributes:
            att.symbolic[JascoCondConst.CRD.value] = SymbolicCondition(frame_chords=torch.tensor(frame_chords))
        return attributes

    @torch.no_grad()
    def _prepare_drums_conditions(self,
                                  attributes:
                                  tp.List[ConditioningAttributes],
                                  drums_wav: tp.Optional[torch.Tensor],
                                  ):
        # prepare drums cond
        for attr in attributes:
            if drums_wav is None:
                attr.wav[JascoCondConst.DRM.value] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
            else:
                if JascoCondConst.DRM.value not in self.lm.condition_provider.conditioners:
                    raise RuntimeError("This model doesn't support drums conditioning. ")

                expected_length = self.lm.cfg.dataset.segment_duration * self.sample_rate
                # trim if needed
                drums_wav = drums_wav[..., :expected_length]

                # pad if needed
                if drums_wav.shape[-1] < expected_length:
                    diff = expected_length - drums_wav.shape[-1]
                    diff_zeros = torch.zeros((drums_wav.shape[0], drums_wav.shape[1], diff),
                                             device=drums_wav.device, dtype=drums_wav.dtype)
                    drums_wav = torch.cat((drums_wav, diff_zeros), dim=-1)

                attr.wav[JascoCondConst.DRM.value] = WavCondition(
                    drums_wav.to(device=self.device),
                    torch.tensor([drums_wav.shape[-1]], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None],
                )

        return attributes

    @torch.no_grad()
    def _prepare_melody_conditions(
            self,
            attributes: tp.List[ConditioningAttributes],
            melody: tp.Optional[torch.Tensor],
            expected_length: int,
            melody_bins: int = 53,
    ) -> tp.List[ConditioningAttributes]:
        """
        Prepares melody conditions by subtituting with pre-computed salience matrix.
        This method updates the ConditioningAttributes with per-frame chords information.
        Args:
            attributes (List[ConditioningAttributes]):
                The initial attributes and optional tensor data.
            chords (List[Tuple[str, float]]):
                A list of tuples containing chord labels and their start times.
        Returns:
            List[ConditioningAttributes]:
                The updated attributes with frame chords integrated, alongside the original optional tensor data.
        """
        for attr in attributes:
            if melody is None:
                melody = torch.zeros((melody_bins, expected_length))
            attr.symbolic[JascoCondConst.MLD.value] = SymbolicCondition(melody=melody)
        return attributes

    @torch.no_grad()
    def _prepare_temporal_conditions(
            self,
            attributes: tp.List[ConditioningAttributes],
            expected_length: int,
            chords: tp.Optional[tp.List[tp.Tuple[str, float]]],
            drums_wav: tp.Optional[torch.Tensor],
            salience_matrix: tp.Optional[torch.Tensor],
            melody_bins: int = 53,
    ) -> tp.List[ConditioningAttributes]:
        """
        Prepares temporal conditions (chords, drums).
        Args:
            attributes (List[ConditioningAttributes]): The initial attributes and optional tensor data.
            expected_length (int): The expected number of generated frames.
            chords (List[Tuple[str, float]]):  A list of tuples containing chord labels and their start times.
            drums_wav (List[Tuple[str, float]]): tensor of extracted drums wav.
            salience_matrix (List[Tuple[str, float]]): melody matrix.
            melody_bins (int): number of melody bins the model was trained with, only relevant if trained with melody.
        Returns:
            List[ConditioningAttributes]:
                The updated attributes after processing chord conditions.
        """
        attributes = self._prepare_chord_conditions(attributes=attributes, chords=chords)
        attributes = self._prepare_drums_conditions(attributes=attributes, drums_wav=drums_wav)
        attributes = self._prepare_melody_conditions(attributes=attributes, melody=salience_matrix,
                                                     expected_length=expected_length, melody_bins=melody_bins)
        return attributes

    @torch.no_grad()
    def generate_music(
        self, descriptions: tp.List[str],
        drums_wav: tp.Optional[torch.Tensor] = None,
        drums_sample_rate: int = 32000,
        chords: tp.Optional[tp.List[tp.Tuple[str, float]]] = None,
        melody_salience_matrix: tp.Optional[torch.Tensor] = None,
        iopaint_wav: tp.Optional[torch.Tensor] = None,
        segment_duration: float = 10.0,
        frame_rate: float = 50.0,
        melody_bins: int = 53,
        progress: bool = False, return_latents: bool = False) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and temporal conditions (chords, melody, drums).

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            chords (list of (str, float) tuples): Chord progression represented as chord, start time (sec), e.g.:
                                                  [("C", 0.0), ("F", 4.0), ("G", 6.0), ("C", 8.0)]
            melody_salience_matrix (torch.Tensor, optional): melody saliency matrix. Default=None.
            iopaint_wav (torch.Tensor, optional): in/out=painting waveform. Default=None.
            segment_duration (float): the segment duration the model was trained on. Default=None.
            frame_rate (float): the frame_rate model was trained on. Default=None.
            melody_bins (int): number of melody bins the model was trained with, only relevant if trained with melody.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """

        if drums_wav is not None:
            if drums_wav.dim() == 2:
                drums_wav = drums_wav[None]
            assert drums_wav.dim() == 3, "drums wav should have a shape [B, C, T]."
            drums_wav = convert_audio(drums_wav, drums_sample_rate, self.sample_rate, self.audio_channels)

        cond_attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions,
                                                                             prompt=None)

        # prepare temporal conds (symbolic / audio)
        jasco_attributes = self._prepare_temporal_conditions(attributes=cond_attributes,
                                                             expected_length=int(segment_duration * frame_rate),
                                                             chords=chords,
                                                             drums_wav=drums_wav,
                                                             salience_matrix=melody_salience_matrix,
                                                             melody_bins=melody_bins)
        assert prompt_tokens is None
        tokens = self._generate_tokens(jasco_attributes, prompt_tokens, progress)
        if return_latents:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    @torch.no_grad()
    def generate(self, descriptions: tp.List[str], progress: bool = False, return_latents: bool = False) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        return self.generate_music(descriptions=descriptions, progress=progress, return_latents=return_latents)
