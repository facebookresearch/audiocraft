# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig
from . import builders, musicgen
from einops import rearrange
from torch.nn import functional as F
from ..modules.conditioners import SegmentWithAttributes

import torch
import numpy as np
import random
import typing as tp
import math
import flashy

import time

def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


class HybridMagnetSolver(musicgen.MusicGenSolver):
    """Solver for MAGNeT - Masked Audio Generation using
        a single Non-autoregressive Transformer https://arxiv.org/abs/2401.04577.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # initialize generation parameters by config
        self.generation_params = {
            'use_sampling': self.cfg.generate.lm.use_sampling,
            'temp': self.cfg.generate.lm.temp,
            'top_k': self.cfg.generate.lm.top_k,
            'top_p': self.cfg.generate.lm.top_p,
            'max_cfg_coef': self.cfg.generate.lm.max_cfg_coef,
            'min_cfg_coef': self.cfg.generate.lm.min_cfg_coef,
            'decoding_steps': list(self.cfg.generate.lm.decoding_steps),
            'anneal_temp': self.cfg.generate.lm.anneal_temp,
            'span_scoring': self.cfg.generate.lm.span_scoring,
            'span_arrangement': self.cfg.generate.lm.span_arrangement
        }

        sequence_len = int(cfg.dataset.segment_duration * self.compression_model.frame_rate)
        # self.mean_maskrate_to_u = torch.tensor(self._calc_mean_maskrate_to_u_LUT(sequence_len), device=self.device)
        # make it a 2D tensor to account any kind of seq_len depending on random prompt duration
        self.mean_maskrate_to_k = torch.cat(
            [
                torch.tensor(
                    self._calc_mean_maskrate_to_u_LUT(seq_len),
                    device=self.device,
                ).unsqueeze(0)
                for seq_len in range(1, sequence_len + 1, 1)
            ],
            dim=0,
        )
        self.ce_per_codebook = [torch.log(torch.tensor(self.compression_model.cardinality, device=self.device))
                                for _ in range(cfg.transformer_lm.n_q)]

    def build_model(self) -> None:
        self.cfg.transformer_lm.segment_duration = self.cfg.dataset.segment_duration
        self.cfg.transformer_lm.span_len = self.cfg.masking.span_len
        assert self.cfg.efficient_attention_backend == "xformers", "MAGNeT v1 models support only xformers backend."
        super().build_model()

    def _calc_mean_maskrate_to_u_LUT(self, T: int):
        """ Create a Look Up Table (LUT) transforming a discrete masking percentage m in 0,1,...,100 to u,
            the number of overlapping spans of length L to place s.t. the masking rate is approximately m/float(100).
            It first creates the inverse transformation, of the masking rate as function of u,
            using the expression choose(T - L, u) / choose(T, u), where L is the atomic span length used
            during masking. See https://arxiv.org/abs/2401.04577,
            appendix C, for the mean mask rate derivation.

            We leverage the fact that:
                                choose(T - L, u) / choose(T, u) = Prod_{j = 0}^{u - 1}((T - L - j)/(T - j))
            in the provided implementation, in order to avoid overflow.
        Args:
            T (float): Sequence length.
        Returns:
            (List) A LUT transforming m in 0,1,...,100 to u,
            s.t. the masking rate of the span-L mask is approximately m/float(100).
        """

        L = self.cfg.masking.span_len

        u2mean = [0.0]  # mean mask rate is 0.0 for u = 0
        v = (T - L) / float(T)
        for u in range(1, T):
            u2mean.append(1 - v)
            v *= (T - L - u) / (T - u)  # Overflow-safe implementation of choose(T - L, u) / choose(T, u).

        mean2u = []
        for maskperc in range(101):
            maskrate = maskperc / float(100)
            u = int(np.searchsorted(u2mean, maskrate))
            mean2u.append(u)

        return mean2u

    def non_spans_mask(self, batch, prompt_timestep, gen_seq_len, rand_mask_probs, device):
        num_token_masked = ((gen_seq_len - prompt_timestep) * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((batch, gen_seq_len), device=device).argsort(
            dim=-1
        ) * (
            torch.arange(gen_seq_len, device=device).repeat(batch, 1) >= prompt_timestep
        )
        return batch_randperm >= gen_seq_len - num_token_masked

    def calc_spans_mask(self, rand_mask_probs, batch, prompt_timestep, gen_seq_len, device):
        rounded_probs = torch.round(100 * rand_mask_probs).long()
        k = self.mean_maskrate_to_k[
            gen_seq_len - prompt_timestep - 1, rounded_probs
        ].clamp(min=1)  # k is the number of span starts

        # sample random span starts
        batch_randperm = torch.rand((batch, gen_seq_len), device=device) * (
            torch.arange(gen_seq_len, device=device).repeat(batch, 1) >= prompt_timestep
        )
        mask = batch_randperm.argsort(dim=-1) >= gen_seq_len - k
        B, T = mask.shape
        shifted_mask = mask.clone()
        for _ in range(self.cfg.masking.span_len - 1):
            shifted_mask = torch.concat((torch.full((B, 1), False, device=device), shifted_mask[:, :-1]), dim=1)
            mask = torch.logical_or(mask, shifted_mask)

        return mask

    def _get_mask(self, mask_probs: torch.Tensor, prompt_timestep: int, B: int, T: int, device: torch.device) -> torch.Tensor:
        """ Construct a boolean mask with masking rates defined by mask_probs, and atomic
            span length defined by cfg.masking.span_len.
        Args:
            mask_probs (torch.Tensor): The desired masking rate per sample, of shape [B,]
            B (int): Batch size.
            T (int): Sequence length.
            device (torch.device): device of the output tensor
        Returns:
            (torch.Tensor): A boolean tensor of shape [B, T]
        """
        if self.cfg.masking.span_len <= 1:
            return self._non_spans_mask(mask_probs, prompt_timestep, B, T, device)

        return self._spans_mask(mask_probs, prompt_timestep, B, T, device)

    def _compute_cross_entropy_hybrid_magnet(self, logits: torch.Tensor,
                                      targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed only on a specific codebook, defined by the stage argument.
        Valid timesteps for each codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
            
        Returns:
            ce (torch.Tensor): Cross entropy of the codebook that is being optimized.
        """
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        logits_k = logits.contiguous().view(-1, logits.size(-1))  # [B x K x T, card]
        targets_k = targets.contiguous().view(-1)  # [B x K x T]
        mask_k = mask.contiguous().view(-1)  # [B x K x T]

        IGNORE_IDX = -1
        targets_k[~mask_k] = IGNORE_IDX
        q_ce = F.cross_entropy(logits_k, targets_k, ignore_index=IGNORE_IDX)

        ce += q_ce
        return ce
    
    def compute_attn_mask(self, decoding_steps):
        delta = decoding_steps.unsqueeze(-1) - decoding_steps.unsqueeze(-2)
        valid = delta >= 0
        return torch.where(
            valid,
            torch.zeros([], device=decoding_steps.device),
            torch.full([], float("-inf"), device=decoding_steps.device),
        ).unsqueeze(1)
    
    def _apply_nar_masks(self, codes: torch.Tensor):
        B, K, T = codes.shape
        device = self.device

        stage = torch.multinomial(torch.full((K,), 1/K, device=device), B, replacement=True)
        stage = stage.unsqueeze(1)  # [B, 1]
        rand_time = uniform((B, 1), device=device)
        rand_mask_probs = torch.cos(rand_time * math.pi * 0.5)

        prompt_timestep = torch.randint(0, T, (B, 1), device=device)
        decoding_steps = prompt_timestep.repeat(1, T + K)

        shifted_prompt_timestep = prompt_timestep - stage  # account for the delay pattern which will predict token levels with a shift
        shifted_prompt_timestep[shifted_prompt_timestep < 0] = 0
        stage_mask = self.get_mask(
            B, shifted_prompt_timestep, T, rand_mask_probs, codes.device
        )
        stage_mask = stage_mask.unsqueeze(1)
        stage = stage.unsqueeze(1)

        prompt_timestep = (
            prompt_timestep - torch.arange(K, device=codes.device).unsqueeze(0)
        ).unsqueeze(-1)

        # mask all tokens after prompt
        timestep_mask = (
            torch.arange(T, device=codes.device).repeat(B, K, 1) >= prompt_timestep
        )
        # mask all RVQ levels above rvq_level
        rvq_levels_mask = (
            torch.arange(K, device=codes.device).repeat(B, 1).unsqueeze(-1) > stage
        )
        # mask for rvq_level
        rvq_level_mask = (
            torch.arange(K, device=codes.device).repeat(B, 1).unsqueeze(-1) == stage
        )
        # mask for input codes
        to_mask = (
            torch.ones_like(codes, dtype=torch.bool) * rvq_levels_mask
            + stage_mask * rvq_level_mask
        ) * timestep_mask

        # assert torch.allclose(
        #     mask_count_now,
        #     torch.sum(to_mask, dim=-1).gather(-1, rvq_level.squeeze(-1)),
        # )

        codes_input = torch.where(
            to_mask,
            torch.full_like(codes, self.model.special_token_id),
            codes,
        )

        # mask for target codes (loss computation)
        loss_mask = (
            torch.ones_like(codes, dtype=torch.bool)
            * timestep_mask
            * stage_mask
            * rvq_level_mask
        )

        prompt_mask = (
            torch.arange(T, device=codes.device).repeat(B, K, 1) < prompt_timestep
        )
        loss_mask += torch.ones_like(codes, dtype=torch.bool) * prompt_mask
        stage_mask = self.get_mask(B, shifted_prompt_timestep, T, rand_mask_probs, device) 
        stage_mask = stage_mask.unsqueeze(1)
        # NAR attention mask
        nar_mask = self.model.attn_mask_per_stage[stage.squeeze(1)]

        # AR attention mask up to prompt_timestep
        seq = torch.arange(T + K, device=device).repeat(B, 1)
        mask = seq < decoding_steps
        decoding_steps = mask * seq + (~mask) * decoding_steps
        ar_mask = self.compute_attn_mask(decoding_steps)

        # semi-AR attention mask
        sar_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(-1).repeat(1, self.model.num_heads, 1, T + K), ar_mask, nar_mask
        )

        return codes_input, loss_mask, sar_mask



    def run_step(self, idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict:
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == 'cuda'

        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
            batch, check_synchronization_points)

        self.deadlock_detect.update('tokens_and_conditions')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('warn')

        codes_input, loss_mask, src_mask = self._apply_nar_masks(audio_tokens)

        with self.autocast:
            model_output = self.model.compute_predictions(codes_input, [], condition_tensors, src_mask=src_mask)
            logits = model_output.logits[:, :, :-self.model.n_q + 1, :]  # ignore extra nan logits after translation to PARALLEL
            loss_mask &= padding_mask
            ce = self._compute_cross_entropy(logits, audio_tokens, loss_mask)
            loss = ce
        self.deadlock_detect.update('loss')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('default')

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

        metrics['ce'] = ce
        metrics['ppl'] = torch.exp(ce)

        return metrics

    @torch.no_grad()
    def run_generate_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
                          gen_duration: float, prompt_duration: tp.Optional[float] = None,
                          remove_prompt: bool = False,
                          **generation_params) -> dict:
        """Run generate step on a batch of optional audio tensor and corresponding attributes.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
            use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
            gen_duration (float): Target audio duration for the generation.
            prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
            remove_prompt (bool, optional): Whether to remove the prompt from the generated audio.
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
        # TODO: Add dropout for chroma?

        # prepare audio prompt
        if prompt_duration is None:
            prompt_audio = None
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_audio_frames = int(prompt_duration * self.compression_model.sample_rate)
            prompt_audio = audio[..., :prompt_audio_frames]

        max_prompt_len_tokens = int(max_prompt_len * (self.compression_model.frame_rate + 1))
        # get audio tokens from compression model
        if prompt_audio is None or prompt_audio.nelement() == 0:
            num_samples = len(attributes)
            # prompt_tokens = None
            scale = None
            # in unprompted scenario, we generate AR for the prompt duration
            prompt_tokens = self.model.ar_generate(
                None,
                attributes,
                num_samples=num_samples,
                max_gen_len=max_prompt_len_tokens,
                use_sampling=True,
                top_p=0.0,
                top_k=250,
                temp=1.0,
            )
        else:
            num_samples = None
            prompt_audio = prompt_audio.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt_audio)
            assert scale is None, "Compression model in MusicGen should not require rescaling."

        # generate by sampling from the LM
        with self.autocast:
            total_gen_len = math.ceil(gen_duration * self.compression_model.frame_rate)
            gen_tokens = self.model.generate(
                prompt_tokens, attributes, max_gen_len=total_gen_len,
                num_samples=num_samples, **self.generation_params)

        # generate audio from tokens
        assert gen_tokens.dim() == 3
        gen_audio = self.compression_model.decode(gen_tokens, None)

        bench_end = time.time()
        gen_outputs = {
            'rtf': (bench_end - bench_start) / gen_duration,
            'ref_audio': audio,
            'gen_audio': gen_audio,
            'gen_tokens': gen_tokens,
            'prompt_audio': prompt_audio,
            'prompt_tokens': prompt_tokens,
        }
        return gen_outputs