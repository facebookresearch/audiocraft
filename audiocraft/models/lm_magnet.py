# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import typing as tp
import torch
import numpy as np

from ..utils import utils
from ..modules.conditioners import (
    ClassifierFreeGuidanceDropout,
    ConditioningAttributes,
    ConditionType,
)
from .lm import LMModel

logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


class MagnetLMModel(LMModel):
    """Transformer-based, non-autoregressive model, operates on multiple streams of audio tokens (MAGNeT).
    Args:
        subcodes_context (int): The number of timesteps attended in the self-attention blocks of codebooks > 0.
                                When set to -1, attention is unrestricted and all timesteps are attended. Defaults to 5.
        compression_model_framerate (int): frame rate of the audio tokenizer.
        segment_duration (int): Sample length in seconds.
        span_len (int): Determines the length of masking spans. This is the minimal length of consecutive masked tokens,
                        for both training and inference. Defaults to 3.
        **kwargs: Additional parameters for the LMModel.
    """
    def __init__(self, subcodes_context: int = 5, compression_model_framerate: int = 50,
                 segment_duration: int = 10, span_len: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.causal = kwargs['causal']
        self.subcodes_context = subcodes_context
        self.span_len = span_len
        self._build_attn_masks(compression_model_framerate=compression_model_framerate,
                               segment_duration=segment_duration,
                               num_heads=kwargs['num_heads'],
                               device=kwargs['device'], dtype=kwargs['dtype'])

    def restricted_context_attn_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Creates a restricted attention mask (local attention map) where the context
           is determined by self.subcodes_context.
        Args:
            seq_len (int): token sequence length.
            device (torch.device): device of the output tensor.
            dtype (torch.dtype): data type of the output tensor.
        Returns:
            torch.Tensor: The restricted attention mask.
        """
        # Return a context restricted non-causal att mask
        queries_pos = torch.arange(seq_len, device=device).view(-1, 1)
        keys_pos = torch.arange(seq_len, device=device).view(1, -1)

        delta = queries_pos - keys_pos
        valid = torch.abs(delta) <= self.subcodes_context
        return torch.where(
            valid,
            torch.zeros([], device=device, dtype=dtype),
            torch.full([], float('-inf'), device=device, dtype=dtype))

    def _stage_attn_mask(self, stage: int, seq_len: int, num_heads: int,
                         device: torch.device, dtype: torch.dtype) -> tp.Optional[torch.Tensor]:
        """Creates a restricted attention mask given the stage (codebook index).
        Args:
            stage (int): The codebook index. Takes values in [0, n_q].
            seq_len (int): Token sequence length.
            num_heads (int): Num transformer attention heads.
            device (torch.device): device of the output tensor.
            dtype (torch.dtype): data type of the output tensor.
        Returns:
            torch.Tensor: Either a restricted attention mask or None if stage attention is unrestricted.
        """
        sa_mask = None

        if stage > 0 and self.subcodes_context > -1:
            # parallel - non-causal - with restricted subcodes context
            sa_mask = self.restricted_context_attn_mask(seq_len, device=device, dtype=dtype)

        if sa_mask is not None:
            # Repeat for each attention head
            sa_mask = sa_mask.repeat((1, num_heads, 1, 1))

            # align8 to enable memory efficient attention
            MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR = 8
            seq_len_aligned = \
                int(np.ceil(seq_len / MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR)) * MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR

            sa_mask_aligned = torch.zeros((1, num_heads, seq_len_aligned, seq_len_aligned), device=device, dtype=dtype)
            sa_mask_aligned[..., :seq_len, :seq_len] = sa_mask
            sa_mask = sa_mask_aligned

        return sa_mask

    def _build_attn_masks(self, compression_model_framerate: int, segment_duration: int, num_heads: int,
                          device: torch.device, dtype: torch.dtype):
        """Construct attention mask per stage. For each of the RVQ codebook levels in the [0, n_q] range,
           either a local attention map or None would be stored as an entry in the self.attn_mask_per_stage list.
        Args:
            compression_model_framerate (int): The frame rate of the tokenizer.
            segment_duration (int): Sample length in seconds.
            num_heads (int): Num transformer attention heads.
            device (torch.device): device of the output tensor.
            dtype (torch.dtype): data type of the output tensor.
        """
        seq_len = compression_model_framerate * segment_duration
        self.attn_mask_per_stage = [self._stage_attn_mask(stage, seq_len, num_heads,
                                                          device, dtype) for stage in range(self.n_q)]

    @torch.no_grad()
    def generate(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 cfg_coef_beta: tp.Optional[float] = None,
                 two_step_cfg: tp.Optional[bool] = None,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 **kwargs) -> torch.Tensor:

        assert cfg_coef is None, "Unsupported in MAGNeT. Use max_cfg_coef,min_cfg_coef instead."
        assert two_step_cfg is None, "MAGNeT currently doesn't support two step classifier-free-guidance."
        assert remove_prompts is False, "MAGNeT currently doesn't support the remove_prompts arg."
        assert check is False, "MAGNeT currently doesn't support the check arg."
        assert cfg_coef_beta is None, "MAGNeT currently doesn't support the cfg_coef_beta arg."
        # Call the MAGNeT-specific generation method
        return self._generate_magnet(prompt=prompt,
                                     conditions=conditions,
                                     num_samples=num_samples,
                                     max_gen_len=max_gen_len,
                                     use_sampling=use_sampling,
                                     temp=temp,
                                     top_k=top_k,
                                     top_p=top_p,
                                     callback=callback, **kwargs)

    @torch.no_grad()
    def _generate_magnet(self,
                         prompt: tp.Optional[torch.Tensor] = None,
                         conditions: tp.List[ConditioningAttributes] = [],
                         num_samples: tp.Optional[int] = None,
                         max_gen_len: int = 256,
                         use_sampling: bool = True,
                         temp: float = 3.0,
                         top_k: int = 0,
                         top_p: float = 0.9,
                         callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                         max_cfg_coef: float = 10.0,
                         min_cfg_coef: float = 1.0,
                         decoding_steps: tp.List[int] = [20, 10, 10, 10],
                         anneal_temp: bool = True,
                         span_scoring='max',
                         span_arrangement='nonoverlap') -> torch.Tensor:
        """Generate audio tokens given textual conditions, and optionally given audio prompts,
        by running MAGNeT's iterative decoding algorithm for each of the n_q RVQ levels.
        Args:
            prompt (torch.Tensor): Prompt tokens of shape [B, K, T].
            conditions (list of ConditioningAttributes): List of conditions.
            num_samples (int): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Initial sampling temperature.
            top_k (int): k for "top-k" sampling.
            top_p (float): p for "top-p" sampling.
            callback (Callback): Callback function to report generation progress.
            max_clsfg_coef (float): Initial coefficient used for classifier free guidance.
            min_clsfg_coef (float): Final coefficient used for classifier free guidance.
            decoding_steps (list of n_q ints): The number of iterative decoding steps,
                                            for each of the n_q RVQ codebooks.
            anneal_temp (bool): When set to True, softmax temperature will be linearly decayed to zero, at each stage.
            span_scoring (str): Use the maximum probability of each span ('max')
                                or the product of probabilities ('prod').
            span_arrangement (str): Use either non-overlapping spans ('nonoverlap') or overlapping spans ('stride1').
                                                in the masking scheme.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistent.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        cfg_conditions: tp.Optional[ConditionTensors]
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            conditions = conditions + null_conditions
            tokenized = self.condition_provider.tokenize(conditions)
            cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, prompt_length = prompt.shape
        start_offset = prompt_length
        assert start_offset < max_gen_len

        mask_id = self.special_token_id

        # we generate codes with a fixed sequence length
        shape = (B, K, max_gen_len)

        gen_codes = torch.full(shape, mask_id, dtype=torch.long, device=device)
        # filling the gen_codes with the prompt if needed
        gen_codes[..., :start_offset] = prompt
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence = gen_codes

        curr_step = 0
        for stage, n_steps in zip(range(self.n_q), decoding_steps):
            gen_sequence, curr_step = self._generate_stage(gen_sequence,
                                                           cfg_conditions,
                                                           stage=stage,
                                                           device=device,
                                                           prompt_length=prompt_length,
                                                           prompt=prompt,
                                                           temp=temp,
                                                           max_cfg_coef=max_cfg_coef,
                                                           min_cfg_coef=min_cfg_coef,
                                                           top_k=top_k,
                                                           top_p=top_p,
                                                           timesteps=n_steps,
                                                           anneal_temp=anneal_temp,
                                                           span_scoring=span_scoring,
                                                           use_sampling=use_sampling,
                                                           span_arrangement=span_arrangement,
                                                           curr_step=curr_step,
                                                           total_steps=sum(decoding_steps),
                                                           callback=callback)

        return gen_sequence

    @torch.no_grad()
    def _generate_stage(self,
                        gen_sequence: torch.Tensor,
                        condition_tensors: tp.Optional[ConditionTensors],
                        stage: int,
                        device: torch.device,
                        prompt_length: int = 0,
                        prompt: tp.Optional[torch.Tensor] = None,
                        use_sampling: bool = True,
                        temp: float = 3.0,
                        max_cfg_coef: float = 10.0,
                        min_cfg_coef: float = 1.0,
                        top_k: int = 0,
                        top_p: float = 0.0,
                        timesteps: int = 10,
                        anneal_temp: bool = True,
                        span_scoring: str = 'max',
                        span_arrangement: str = 'nonoverlap',
                        curr_step: int = 0,
                        total_steps: int = 0,
                        callback: tp.Optional[tp.Callable[[int, int], None]] = None) -> tp.Tuple[torch.Tensor, int]:
        """Generate audio tokens of a single RVQ level (stage), given the previously generated stages,
           and the textual conditions.
        Args:
            gen_sequence (torch.Tensor): Previously generated tokens.
            condition_tensors (tp.Optional[ConditionTensors]): pre-computed conditioning tensors.
            stage (int): RVQ level to generate.
            device (torch.device): device of the output tensor.
            prompt_length (int): Temporal length of the audio prompt.
            prompt (torch.Tensor): Prompt tokens of shape [B, K, T].
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Initial sampling temperature.
            max_clsfg_coef (float): Initial coefficient used for classifier free guidance.
            min_clsfg_coef (float): Final coefficient used for classifier free guidance.
            top_k (int): k for "top-k" sampling.
            top_p (float): p for "top-p" sampling.
            timesteps (int): Number of iterative decoding steps.
            anneal_temp (bool): When set to True, softmax temperature will be linearly decayed to zero, at each stage.
            span_scoring (str): Use the maximum probability of each span ('max')
                                or the product of probabilities ('prod').
            span_arrangement (str): Use either non-overlapping spans ('nonoverlap') or overlapping spans ('stride1').
                                                in the masking scheme.
            curr_step (int): Global iterative decoding step counter.
            total_steps (int): Total decoding steps.
            callback (Callback): Callback function to report generation progress.
        Returns:
            tuple(torch.Tensor, int): Generated tokens and the current decoding step counter.
        """
        B, K, T = gen_sequence.shape
        shape = (B, 1, T)  # generating a single codebook per stage

        mask_id = self.special_token_id
        stage_gen_seq = torch.full(shape, mask_id, dtype=torch.long, device=device)

        assert span_arrangement == 'nonoverlap' or span_arrangement == 'stride1'
        chunk_masking = self.span_len > 1 and span_arrangement == 'nonoverlap'

        DONT_REMASK_ME_SCORE = -1e4

        model = self if self._fsdp is None else self._fsdp

        if chunk_masking:
            # span-wise scores
            n_chunks = T // self.span_len
            if T % self.span_len != 0:
                # trim sequence ending to achieve a multiple of span_len
                T = self.span_len * n_chunks
                gen_sequence = gen_sequence[..., :T]
                stage_gen_seq = stage_gen_seq[..., :T]

            chunked_shape = (B, 1, n_chunks)
            n_prompt_chunks = prompt_length // self.span_len
            scores = torch.zeros(chunked_shape, dtype=torch.float32, device=device)
            scores[..., :n_prompt_chunks] = DONT_REMASK_ME_SCORE
            num_chunks_to_gen = n_chunks - n_prompt_chunks
        else:
            # token-wise scores
            scores = torch.zeros(shape, dtype=torch.float32, device=device)
            scores[..., :prompt_length] = DONT_REMASK_ME_SCORE
            gen_T = T - prompt_length

        # run MAGNeT iterative decoding for "timesteps" iterations
        for timestep, steps_left in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):

            mask_p = torch.cos(timestep * math.pi * 0.5)

            if chunk_masking:
                num_masked = max(int((mask_p * num_chunks_to_gen).item()), 1)
            else:
                num_masked = max(int((mask_p * gen_T).item()), 1)

            # masking
            run_lps_masking = (span_arrangement == 'stride1') and self.span_len > 1
            if run_lps_masking:
                # masking of the k least probable overlapping (stride 1) spans
                mask = torch.concat((
                    [self._least_probable_span_masking(scores[[i], :, :], num_masked).to(device)
                     for i in range(B)]), dim=0)
                stage_gen_seq[mask] = mask_id
            else:
                # masking of the k least probable non-overlapping spans
                masked = scores.topk(num_masked, dim=-1).indices
                if chunk_masking:
                    chunks_mask = torch.full(chunked_shape, False, dtype=torch.bool, device=device)
                    chunks_mask = chunks_mask.scatter(2, masked, True)
                    mask = torch.repeat_interleave(chunks_mask, self.span_len, dim=-1)
                    stage_gen_seq[mask] = mask_id
                else:
                    stage_gen_seq = stage_gen_seq.scatter(2, masked, mask_id)

            if prompt is not None:
                stage_gen_seq[..., :prompt_length] = prompt[:, stage, :].unsqueeze(1)

            gen_sequence[:, [stage], :] = stage_gen_seq
            if condition_tensors:
                # duplicate input for classifier free guidance
                sequence = torch.cat([gen_sequence, gen_sequence], dim=0)

            all_logits = model(sequence, [], condition_tensors, stage=stage)

            if condition_tensors:
                # classifier free guidance with annealing
                cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
                clsfg_coef = float(mask_p) * max_cfg_coef + (1 - float(mask_p)) * min_cfg_coef
                logits = uncond_logits + (cond_logits - uncond_logits) * clsfg_coef
            else:
                logits = all_logits

            # temperature annealing - linear
            t = temp * (steps_left / timesteps) if anneal_temp else temp

            # sampling
            logits = logits[:, stage, :, :].unsqueeze(1)
            probs = torch.softmax(logits / max(t, 1e-2), dim=-1)
            if use_sampling:
                if top_p > 0.0:
                    sampled_tokens = utils.sample_top_p(probs, p=top_p)
                elif top_k > 0:
                    sampled_tokens = utils.sample_top_k(probs, k=top_k)
                else:
                    sampled_tokens = utils.multinomial(probs, num_samples=1)
            else:
                sampled_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # place mask_id token in each of the masked positions
            mask = stage_gen_seq == mask_id
            stage_gen_seq = torch.where(mask, sampled_tokens[..., 0], stage_gen_seq)
            gen_sequence[:, [stage], :] = stage_gen_seq

            # get probs of sampled tokens
            sampled_probs = torch.gather(probs, 3, sampled_tokens)[..., 0]

            # span scoring
            if chunk_masking:
                if span_scoring == 'max':
                    # max in linear space
                    scores = 1 - torch.max(sampled_probs.reshape((B, 1, n_chunks, -1)), dim=-1)[0]
                elif span_scoring == 'prod':
                    # prod in log space
                    scores = torch.sum(-torch.log(sampled_probs).reshape((B, 1, n_chunks, -1)), dim=-1)
                else:
                    raise NotImplementedError
            else:
                # prod in log space for lps masking (stride1)
                scores = -torch.log(sampled_probs)

            # Fix unmasked tokens by placing inf probs (-inf scores)
            if chunk_masking:
                scores = scores.masked_fill(~chunks_mask, DONT_REMASK_ME_SCORE)
            else:
                scores = scores.masked_fill(~mask, DONT_REMASK_ME_SCORE)

            if callback is not None:
                curr_step += 1
                callback(curr_step, total_steps)

        return gen_sequence, curr_step

    def _construct_spans_mask(self, span_starts: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
        """Build a [1x1xT] boolean mask consists of overlapping spans of True values, where
           span_starts defines the initial index of each span, and the span length is
           defined by self.span_len.
        Args:
            span_starts (torch.Tensor): Boolean mask determines the temporal location of each span start.
            T (int): Sequence length.
            device (torch.device): device of the output tensor.
        Returns:
            torch.Tensor: Spans mask of shape [1x1xT]
        """
        mask = torch.full((1, 1, T), False, device=device)
        mask[:, :, span_starts] = True
        shifted_mask = mask.clone()
        for _ in range(self.span_len - 1):
            shifted_mask = torch.concat((torch.full((1, 1, 1), False, device=device), shifted_mask[:, :, :-1]), dim=-1)
            mask = torch.logical_or(mask, shifted_mask)
        return mask

    def _least_probable_span_masking(self, scores: torch.Tensor, num_masked_trg: int) -> torch.Tensor:
        """Construct a [1x1xT] boolean mask, consists of the u least probable spans,
           where the token probability is determined by -scores, and the total
           number of masked tokens is as closest as possible to num_masked_trg.
           Find u using binary search.
        Args:
            scores (torch.Tensor): Per token score [-log(prob)]
            num_masked_trg: int: The desired amount of tokens to be masked.
        Returns:
            torch.Tensor: Spans mask of shape [1x1xT]
        """
        T = scores.shape[-1]
        device = scores.device
        scores_unfolded = scores.unfold(2, self.span_len, 1)
        # Span score is the product of probs (sum in log space)
        span_scores = scores_unfolded.sum(dim=-1)
        spans_by_scores = torch.argsort(span_scores[0, 0], descending=True)

        num_masked_trg = max(num_masked_trg, self.span_len)

        # Binary search for u - the number least probable overlapping masked spans s.t.
        # the total masking rate is the closest to num_masked_trg / T.
        min_u = num_masked_trg // self.span_len
        max_u = num_masked_trg - self.span_len + 1
        mid = round(0.5 * (min_u + max_u))

        if mid == min_u or mid == max_u:
            return self._construct_spans_mask(spans_by_scores[:mid], T, device)

        while mid > min_u and mid < max_u:
            mask = self._construct_spans_mask(spans_by_scores[:mid], T, device)
            n_masked = mask.sum()
            if n_masked > num_masked_trg:
                max_u = mid
                mid = round(0.5 * (min_u + max_u))
            else:
                min_u = mid
                mid = round(0.5 * (min_u + max_u))

        return mask
