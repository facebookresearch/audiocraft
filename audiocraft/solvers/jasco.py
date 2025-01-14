# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig
from . import builders, musicgen
from .compression import CompressionSolver
from .. import models
from ..modules.conditioners import JascoCondConst, SegmentWithAttributes
import torch
import typing as tp
import flashy
import time
import math


class JascoSolver(musicgen.MusicGenSolver):
    """Solver for JASCO - Joint Audio and Symbolic Conditioning for Temporally Controlled Text-to-Music Generation
        https://arxiv.org/abs/2406.10970.
    """
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.JASCO

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # initialize generation parameters by config
        self.generation_params = {
            'cfg_coef_all': self.cfg.generate.lm.cfg_coef_all,
            'cfg_coef_txt': self.cfg.generate.lm.cfg_coef_txt
        }

        self.latent_mean = cfg.compression_model_latent_mean
        self.latent_std = cfg.compression_model_latent_std
        self.mse = torch.nn.MSELoss(reduction='none')
        self._best_metric_name = 'loss'

    def build_model(self) -> None:
        """Instantiate model and optimization."""
        assert self.cfg.efficient_attention_backend == "xformers", "JASCO v1 models support only xformers backend."

        self.compression_model = CompressionSolver.wrapped_model_from_checkpoint(
            self.cfg, self.cfg.compression_model_checkpoint, device=self.device)
        assert self.compression_model.sample_rate == self.cfg.sample_rate, (
            f"Compression model sample rate is {self.compression_model.sample_rate} but "
            f"Solver sample rate is {self.cfg.sample_rate}."
            )
        # instantiate JASCO model
        self.model: models.FlowMatchingModel = models.builders.get_jasco_model(self.cfg,
                                                                               self.compression_model).to(self.device)
        # initialize optimization
        self.initialize_optimization()

    def _get_latents(self, audio):
        with torch.no_grad():
            latents = self.compression_model.model.encoder(audio)
        return latents.permute(0, 2, 1)  # [B, D, T] -> [B, T, D]

    def _prepare_latents_and_attributes(
        self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
    ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        audio, infos = batch
        audio = audio.to(self.device)
        assert audio.size(0) == len(infos), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(infos)})"
        )

        latents = self._get_latents(audio)

        # prepare attributes
        if JascoCondConst.CRD.value in self.cfg.conditioners:
            null_chord_idx = self.cfg.conditioners.chords.chords_emb.card
        else:
            null_chord_idx = -1
        attributes = [info.to_condition_attributes() for info in infos]
        if self.model.cfg_dropout is not None:
            attributes = self.model.cfg_dropout(samples=attributes,
                                                cond_types=["wav", "text", "symbolic"],
                                                null_chord_idx=null_chord_idx)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(latents, dtype=torch.bool, device=latents.device)

        return condition_tensors, latents, padding_mask

    def _normalized_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize latents."""
        return (latents - self.latent_mean) / self.latent_std

    def _unnormalized_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Unnormalize latents."""
        return (latents * self.latent_std) + self.latent_mean

    def _z(self, z_0: torch.Tensor, z_1: torch.Tensor, t: torch.Tensor, sigma_min: float = 1e-5) -> torch.Tensor:
        """Interpolate data and prior."""
        return (1 - (1 - sigma_min) * t) * z_0 + t * z_1

    def _vector_field(self, z_0: torch.Tensor, z_1: torch.Tensor, sigma_min: float = 1e-5) -> torch.Tensor:
        """Compute the GT vector field.
           sigma_min is a small value to avoid numerical instabilities."""
        return z_1 - (1 - sigma_min) * z_0

    def _compute_loss(self, t: torch.Tensor, v_theta: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute the loss."""
        loss_func = self.cfg.get('loss_func', 'increasing')
        if loss_func == 'uniform':
            scales = 1
        elif loss_func == 'increasing':
            scales = 1 + t  # type: ignore
        elif loss_func == 'decreasing':
            scales = 2 - t  # type: ignore
        else:
            raise ValueError('unsupported loss_func was passed in config')
        return (scales * self.mse(v_theta, v)).mean()

    def run_step(self, idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict:
        """Perform one training or valid step on a given batch."""

        condition_tensors, latents, padding_mask = self._prepare_latents_and_attributes(batch)

        self.deadlock_detect.update('tokens_and_conditions')

        B, T, D = latents.shape
        device = self.device

        # normalize latents
        z_1 = self._normalized_latents(latents)

        # sample the N(0,1) prior
        z_0 = torch.randn(B, T, D, device=device)

        # random time parameter, between 0 to 1
        t = torch.rand((B, 1, 1), device=device)

        # interpolate data and prior
        z = self._z(z_0, z_1, t)

        # compute the GT vector field
        v = self._vector_field(z_0, z_1)

        with self.autocast:
            v_theta = self.model(latents=z,
                                 t=t,
                                 conditions=[],
                                 condition_tensors=condition_tensors)

            loss = self._compute_loss(t, v_theta, v)
            unscaled_loss = loss.clone()

        self.deadlock_detect.update('loss')

        if self.is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            self.deadlock_detect.update('scale')
            if self.cfg.fsdp.use:
                loss.backward()
                flashy.distrib.average_tensors(self.model.buffers())
            elif self.cfg.optim.eager_sync:
                with flashy.distrib.eager_sync_model(self.model):
                    loss.backward()
            else:
                # this should always be slower but can be useful
                # for weird use cases like multiple backwards.
                loss.backward()
                flashy.distrib.sync_model(self.model)
            self.deadlock_detect.update('backward')

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.max_norm:
                if self.cfg.fsdp.use:
                    metrics['grad_norm'] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
                else:
                    metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.max_norm
                    )
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.deadlock_detect.update('optim')
            if self.scaler is not None:
                scale = self.scaler.get_scale()
                metrics['grad_scale'] = scale
            if not loss.isfinite().all():
                raise RuntimeError("Model probably diverged.")

        metrics['loss'] = unscaled_loss

        return metrics

    def _decode_latents(self, latents):
        return self.compression_model.model.decoder(latents.permute(0, 2, 1))

    @torch.no_grad()
    def run_generate_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
                          gen_duration: float, prompt_duration: tp.Optional[float] = None,
                          remove_text_conditioning: bool = False,
                          **generation_params) -> dict:
        """Run generate step on a batch of optional audio tensor and corresponding attributes.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
            use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
            gen_duration (float): Target audio duration for the generation.
            prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
            remove_text_conditioning (bool, optional): Whether to remove the prompt from the generated audio.
            generation_params: Additional generation parameters.
        Returns:
            gen_outputs (dict): Generation outputs, consisting in audio, audio tokens from both the generation
                and the prompt along with additional information.
        """
        bench_start = time.time()
        audio, meta = batch
        assert audio.size(0) == len(meta), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(meta)})"
        )
        # prepare attributes
        attributes = [x.to_condition_attributes() for x in meta]

        # prepare audio prompt
        if prompt_duration is None:
            prompt_audio = None
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_audio_frames = int(prompt_duration * self.compression_model.sample_rate)
            prompt_audio = audio[..., :prompt_audio_frames]

        # get audio tokens from compression model
        if prompt_audio is None or prompt_audio.nelement() == 0:
            num_samples = len(attributes)
            prompt_tokens = None
        else:
            num_samples = None
            prompt_audio = prompt_audio.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt_audio)
            assert scale is None, "Compression model in MusicGen should not require rescaling."

        # generate by sampling from the LM
        with self.autocast:
            total_gen_len = math.ceil(gen_duration * self.compression_model.frame_rate)
            gen_latents = self.model.generate(
                prompt_tokens, attributes, max_gen_len=total_gen_len,
                num_samples=num_samples, **self.generation_params)

        # generate audio from latents
        assert gen_latents.dim() == 3  # [B, T, D]

        # unnormalize latents
        gen_latents = self._unnormalized_latents(gen_latents)
        gen_audio = self._decode_latents(gen_latents)

        bench_end = time.time()
        gen_outputs = {
            'rtf': (bench_end - bench_start) / gen_duration,
            'ref_audio': audio,
            'gen_audio': gen_audio,
            'gen_tokens': gen_latents,
            'prompt_audio': prompt_audio,
            'prompt_tokens': prompt_tokens,
        }
        return gen_outputs
