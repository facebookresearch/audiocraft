# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi Band Diffusion models as described in
"From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion"
(paper link).
"""

import typing as tp

import torch
import julius

from .unet import DiffusionUnet
from ..modules.diffusion_schedule import NoiseSchedule
from .encodec import CompressionModel
from ..solvers.compression import CompressionSolver
from .loaders import load_compression_model, load_diffusion_models


class DiffusionProcess:
    """Sampling for a diffusion Model.

    Args:
        model (DiffusionUnet): Diffusion U-Net model.
        noise_schedule (NoiseSchedule): Noise schedule for diffusion process.
    """
    def __init__(self, model: DiffusionUnet, noise_schedule: NoiseSchedule) -> None:
        self.model = model
        self.schedule = noise_schedule

    def generate(self, condition: torch.Tensor, initial_noise: torch.Tensor,
                 step_list: tp.Optional[tp.List[int]] = None):
        """Perform one diffusion process to generate one of the bands.

        Args:
            condition (torch.Tensor): The embeddings from the compression model.
            initial_noise (torch.Tensor): The initial noise to start the process.
        """
        return self.schedule.generate_subsampled(model=self.model, initial=initial_noise, step_list=step_list,
                                                 condition=condition)


class MultiBandDiffusion:
    """Sample from multiple diffusion models.

    Args:
        DPs (list of DiffusionProcess): Diffusion processes.
        codec_model (CompressionModel): Underlying compression model used to obtain discrete tokens.
    """
    def __init__(self, DPs: tp.List[DiffusionProcess], codec_model: CompressionModel) -> None:
        self.DPs = DPs
        self.codec_model = codec_model
        self.device = next(self.codec_model.parameters()).device

    @property
    def sample_rate(self) -> int:
        return self.codec_model.sample_rate

    @staticmethod
    def get_mbd_musicgen(device=None):
        """Load our diffusion models trained for MusicGen."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        path = 'facebook/multiband-diffusion'
        filename = 'mbd_musicgen_32khz.th'
        name = 'facebook/musicgen-small'
        codec_model = load_compression_model(name, device=device)
        models, processors, cfgs = load_diffusion_models(path, filename=filename, device=device)
        DPs = []
        for i in range(len(models)):
            schedule = NoiseSchedule(**cfgs[i].schedule, sample_processor=processors[i], device=device)
            DPs.append(DiffusionProcess(model=models[i], noise_schedule=schedule))
        return MultiBandDiffusion(DPs=DPs, codec_model=codec_model)

    @staticmethod
    def get_mbd_24khz(bw: float = 3.0,
                      device: tp.Optional[tp.Union[torch.device, str]] = None,
                      n_q: tp.Optional[int] = None):
        """Get the pretrained Models for MultibandDiffusion.

        Args:
            bw (float): Bandwidth of the compression model.
            device (torch.device or str, optional): Device on which the models are loaded.
            n_q (int, optional): Number of quantizers to use within the compression model.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert bw in [1.5, 3.0, 6.0], f"bandwidth {bw} not available"
        if n_q is not None:
            assert n_q in [2, 4, 8]
            assert {1.5: 2, 3.0: 4, 6.0: 8}[bw] == n_q, \
                f"bandwidth and number of codebooks missmatch to use n_q = {n_q} bw should be {n_q * (1.5 / 2)}"
        n_q = {1.5: 2, 3.0: 4, 6.0: 8}[bw]
        codec_model = CompressionSolver.model_from_checkpoint(
            '//pretrained/facebook/encodec_24khz', device=device)
        codec_model.set_num_codebooks(n_q)
        codec_model = codec_model.to(device)
        path = 'facebook/multiband-diffusion'
        filename = f'mbd_comp_{n_q}.pt'
        models, processors, cfgs = load_diffusion_models(path, filename=filename, device=device)
        DPs = []
        for i in range(len(models)):
            schedule = NoiseSchedule(**cfgs[i].schedule, sample_processor=processors[i], device=device)
            DPs.append(DiffusionProcess(model=models[i], noise_schedule=schedule))
        return MultiBandDiffusion(DPs=DPs, codec_model=codec_model)

    @torch.no_grad()
    def get_condition(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Get the conditioning (i.e. latent representations of the compression model) from a waveform.
        Args:
            wav (torch.Tensor): The audio that we want to extract the conditioning from.
            sample_rate (int): Sample rate of the audio."""
        if sample_rate != self.sample_rate:
            wav = julius.resample_frac(wav, sample_rate, self.sample_rate)
        codes, scale = self.codec_model.encode(wav)
        assert scale is None, "Scaled compression models not supported."
        emb = self.get_emb(codes)
        return emb

    @torch.no_grad()
    def get_emb(self, codes: torch.Tensor):
        """Get latent representation from the discrete codes.
        Args:
            codes (torch.Tensor): Discrete tokens."""
        emb = self.codec_model.decode_latent(codes)
        return emb

    def generate(self, emb: torch.Tensor, size: tp.Optional[torch.Size] = None,
                 step_list: tp.Optional[tp.List[int]] = None):
        """Generate waveform audio from the latent embeddings of the compression model.
        Args:
            emb (torch.Tensor): Conditioning embeddings
            size (None, torch.Size): Size of the output
                if None this is computed from the typical upsampling of the model.
            step_list (list[int], optional): list of Markov chain steps, defaults to 50 linearly spaced step.
        """
        if size is None:
            upsampling = int(self.codec_model.sample_rate / self.codec_model.frame_rate)
            size = torch.Size([emb.size(0), self.codec_model.channels, emb.size(-1) * upsampling])
        assert size[0] == emb.size(0)
        out = torch.zeros(size).to(self.device)
        for DP in self.DPs:
            out += DP.generate(condition=emb, step_list=step_list, initial_noise=torch.randn_like(out))
        return out

    def re_eq(self, wav: torch.Tensor, ref: torch.Tensor, n_bands: int = 32, strictness: float = 1):
        """Match the eq to the encodec output by matching the standard deviation of some frequency bands.
        Args:
            wav (torch.Tensor): Audio to equalize.
            ref (torch.Tensor): Reference audio from which we match the spectrogram.
            n_bands (int): Number of bands of the eq.
            strictness (float): How strict the matching. 0 is no matching, 1 is exact matching.
        """
        split = julius.SplitBands(n_bands=n_bands, sample_rate=self.codec_model.sample_rate).to(wav.device)
        bands = split(wav)
        bands_ref = split(ref)
        out = torch.zeros_like(ref)
        for i in range(n_bands):
            out += bands[i] * (bands_ref[i].std() / bands[i].std()) ** strictness
        return out

    def regenerate(self, wav: torch.Tensor, sample_rate: int):
        """Regenerate a waveform through compression and diffusion regeneration.
        Args:
            wav (torch.Tensor): Original 'ground truth' audio.
            sample_rate (int): Sample rate of the input (and output) wav.
        """
        if sample_rate != self.codec_model.sample_rate:
            wav = julius.resample_frac(wav, sample_rate, self.codec_model.sample_rate)
        emb = self.get_condition(wav, sample_rate=self.codec_model.sample_rate)
        size = wav.size()
        out = self.generate(emb, size=size)
        if sample_rate != self.codec_model.sample_rate:
            out = julius.resample_frac(out, self.codec_model.sample_rate, sample_rate)
        return out

    def tokens_to_wav(self, tokens: torch.Tensor, n_bands: int = 32):
        """Generate Waveform audio with diffusion from the discrete codes.
        Args:
            tokens (torch.Tensor): Discrete codes.
            n_bands (int): Bands for the eq matching.
        """
        wav_encodec = self.codec_model.decode(tokens)
        condition = self.get_emb(tokens)
        wav_diffusion = self.generate(emb=condition, size=wav_encodec.size())
        return self.re_eq(wav=wav_diffusion, ref=wav_encodec, n_bands=n_bands)
