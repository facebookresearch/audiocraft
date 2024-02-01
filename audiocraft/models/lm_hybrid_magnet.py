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
from .lm import LMModel, LMOutput

logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]

MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR = 8


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
                         device: torch.device, dtype: torch.dtype, mem_eff_seq_len: int = None) -> tp.Optional[torch.Tensor]:

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
        
        else:
            sa_mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)

        if mem_eff_seq_len is not None:
            seq_len_aligned = mem_eff_seq_len
        else:
            # sequence will be longer by self.n_q positions: 1 on the left for the AR input, self.n_q - 1 on the right to account for the shifted higher levels
            seq_len_aligned = seq_len+self.n_q
        assert seq_len_aligned % MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR == 0  # comment this for integ tests

        final_mask = torch.full((seq_len_aligned, seq_len_aligned),
                                float('-inf'),
                                device=device, dtype=dtype)

        # shift sa_mask based on stage to account for the delay pattern shift
        # columns (keys) are shifted one more time to account for the left padded input
        # for the rows we match the AR targets positions (generated time step t, stage k will be at index t + k)
        final_mask[stage:stage+seq_len, 1+stage:1+stage+seq_len] = sa_mask
        # placeholder to avoid nans, but we will just ignore what happens after stage+seq_len for loss computation
        if stage+seq_len < seq_len_aligned:
            final_mask[stage+seq_len:, :] = 0
        # placeholder to avoid nans, but in practice it will be overwritten by AR-causal mask for joint training, and ignored during inference
        if stage > 0:
            final_mask[:stage, :] = 0

        # if final_mask is not None:
        #     # align8 to enable memory efficient attention
        #     MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR = 8
        #     seq_len_aligned = int(np.ceil((delay_seq_len) / MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR)) * MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR
        #     final_mask_aligned = torch.zeros((seq_len_aligned, seq_len_aligned), device=device, dtype=dtype)
        #     final_mask_aligned[:delay_seq_len, :delay_seq_len] = final_mask
        #     final_mask = final_mask_aligned

        return final_mask
    
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
        # self.attn_mask_per_stage = [self._stage_attn_mask(stage, seq_len, num_heads,
        #                                                   device, dtype) for stage in range(self.n_q)]

        self.attn_mask_per_stage = torch.cat([self._stage_attn_mask(stage, seq_len, device, dtype).unsqueeze(0)
                                              for stage in range(self.n_q)], dim=0)
    
    def forward(self, sequence: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                # directly pass attn_mask to work with randomized stages
                src_mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, K, S] with K the number of codebooks and
        S the sequence steps, return the logits with shape [B, card, K, S].

        Args:
            indices (torch.Tensor): Indices of the codes to model.
            conditions (list of ConditioningAttributes): Conditions to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see `conditions`.
            stage (int): The codebook level that is being predicted. Relevant for MAGNeT
                in which prediction is done in a codebook-by-codebook manner.
                Takes values in range(n_q), and ignored by default.
        Returns:
            torch.Tensor: Logits.
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            # encode conditions and fuse, both have a streaming cache to not recompute when generating.
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        out = self.transformer(input_, cross_attention_src=cross_attention_input,
                               src_mask=src_mask)
        if self.out_norm:
            out = self.out_norm(out)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, S, card]

        # remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]

        return logits  # [B, K, S, card]
    
    def compute_predictions(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None,
            # directly pass attn_mask to work with randomized stages
            src_mask: tp.Optional[torch.Tensor] = None,
            keep_only_valid_steps: bool = True) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list of ConditioningAttributes): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning
                tensors, see `conditions`.
            stage (int): The codebook level that is being predicted. Relevant for MAGNeT
                in which prediction is done in a codebook-by-codebook manner.
                Takes values in range(n_q), and ignored by default.
            keep_only_valid_steps (bool): Build a sequence from the pattern up to valid (= fully defined) steps.
                Steps that are beyond valid steps will be replaced by the special_token in that case.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        B, K, T = codes.shape
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        # we want to use DELAY and keep higher level tokens at the end of sequence
        pattern = pattern = self.pattern_provider.get_pattern(T+K-1)  
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=keep_only_valid_steps,
        )

        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, conditions, condition_tensors, src_mask=src_mask)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=keep_only_valid_steps
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        return LMOutput(logits, logits_mask)
    
    @torch.no_grad()
    def ar_generate(self,  # copied from main branch for AR prompt generation
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = False,
                 temp: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.0,
                 remove_prompts: bool = False) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.
        Args:
            prompt (Optional[torch.Tensor]): Prompt tokens of shape [B, K, T].
            conditions (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): Set of conditions.
            num_samples (int or None): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            remove_prompts (bool): Whether to remove prompts from generation or not.
            profiler (Profiler or None): profiler to use, it must be already activated.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistents.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsitent inputs shapes"
        num_samples = possible_num_samples[0]

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we than do 1 forward pass instead of 2.
        # the reason for that is two-fold:
        # 1. it is about x2 faster than doing 2 forward passes
        # 2. avoid the streaming API treating the 2 passes as part of different time steps
        condition_tensors: tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]
        if conditions:
            conditions = self.att_dropout(conditions)
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if self.two_step_cfg:
                condition_tensors = (
                    self.condition_provider(self.condition_provider.tokenize(conditions)),
                    self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                )
            else:
                conditions = conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions)
                condition_tensors = self.condition_provider(tokenized)
        else:
            condition_tensors = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        assert start_offset < max_gen_len

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        # filling the gen_codes with the prompt if needed
        gen_codes[..., :start_offset] = prompt
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        with self.streaming():
            self._uncond_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
            for offset in range(start_offset_sequence, gen_sequence_len):
                # get current sequence (note that the streaming API is providing the caching over previous offsets)
                curr_sequence = gen_sequence[..., prev_offset:offset]
                curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                # check coherence between mask and sequence
                assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                # should never happen as gen_sequence is filled progressively
                assert not (curr_sequence == unknown_token).any()
                # sample next token from the model, next token shape is [B, K, 1]
                next_token = self._sample_next_token(
                    curr_sequence, condition_tensors, use_sampling, temp, top_k, top_p)
                # ensure the tokens that should be masked are properly set to special_token_id
                # as the model never output special_token_id
                valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                next_token[~valid_mask] = self.special_token_id
                # ensure we don't overwrite prompt tokens, we only write over unknown tokens
                # (then mask tokens should be left as is as well, which is correct)
                gen_sequence[..., offset:offset+1] = torch.where(
                    gen_sequence[..., offset:offset+1] == unknown_token,
                    next_token, gen_sequence[..., offset:offset+1]
                )
                prev_offset = offset
                
        self._uncond_state.clear()

        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes
    
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
                 two_step_cfg: tp.Optional[bool] = None,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 **kwargs) -> torch.Tensor:

        assert cfg_coef is None, "Unsupported in MAGNeT. Use max_cfg_coef,min_cfg_coef instead."
        assert two_step_cfg is None, "MAGNeT currently doesn't support two step classifier-free-guidance."
        assert remove_prompts is False, "MAGNeT currently doesn't support the remove_prompts arg."
        assert check is False, "MAGNeT currently doesn't support the check arg."
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

        init_T = max_gen_len
        mem_eff_T = -1
        
        # we generate codes with a fixed sequence length
        # shape = (B, K, max_gen_len)
        span_len_T = int(np.ceil(init_T / self.span_len)) * self.span_len
        while mem_eff_T != span_len_T:
            mem_eff_T = int(np.ceil((span_len_T+self.n_q) / MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR)) * MEMORY_EFFICIENT_ATTN_ALIGN_FACTOR - self.n_q
            span_len_T = int(np.ceil(mem_eff_T / self.span_len)) * self.span_len
        if self.attn_mask_per_stage.shape[-1] != mem_eff_T:
            self.attn_mask_per_stage = torch.cat([self._stage_attn_mask(stage, init_T, device, self.attn_mask_per_stage.dtype, mem_eff_seq_len=mem_eff_T+self.n_q).unsqueeze(0) for stage in range(self.n_q)], dim=0)
        shape = (B, K, mem_eff_T)


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

        # return gen_sequence
        return gen_sequence[:, :, :init_T]

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

            assert T % self.span_len == 0, "T % span_len != 0, will not work with custom attention mask dimensions"

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
            else:
                sequence = gen_sequence

            # all_logits = model(sequence, [], condition_tensors, stage=stage)
            all_logits = model(sequence, [], condition_tensors, src_mask=self.attn_mask_per_stage[stage])

            if condition_tensors:
                # classifier free guidance with annealing
                cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
                clsfg_coef = float(mask_p) * max_cfg_coef + (1 - float(mask_p)) * min_cfg_coef
                logits = uncond_logits + (cond_logits - uncond_logits) * clsfg_coef
            else:
                logits = all_logits

            # temperature annealing - linear
            t = temp * (steps_left / timesteps) if anneal_temp else temp

            # ignore the nan logits added after translation to PARALLEL
            logits = logits[:, stage, :-K+1, :].unsqueeze(1)

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
