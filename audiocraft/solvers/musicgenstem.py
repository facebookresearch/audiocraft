# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import time
import typing as tp
import warnings

import flashy
import math
import omegaconf
import random
import torch
from torch.nn import functional as F

from . import base, builders
from .compression import CompressionSolver
from .. import metrics as eval_metrics
from .. import models
from ..data.audio_dataset import AudioDataset
from ..data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from ..data.audio_utils import normalize_audio
from ..modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, AttributeDropout, MultiStemStyleConditioner
from ..utils.cache import CachedBatchWriter, CachedBatchLoader
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_dataset_from_loader, is_jsonable, warn_once, model_hash


class MusicGenStemSolver(base.StandardSolver):
    """Solver for MusicGen-Stem training task.

    Used in: TODO: add arxiv link to MusicGen-Stem
    """
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.MUSIC

    def __init__(self, cfg: omegaconf.DictConfig):
        from demucs import pretrained
        super().__init__(cfg)
        # easier access to sampling parameters
        self.generation_params = {
            'use_sampling': self.cfg.generate.lm.use_sampling,
            'temp': self.cfg.generate.lm.temp,
            'top_k': self.cfg.generate.lm.top_k,
            'top_p': self.cfg.generate.lm.top_p,
        }
        self._best_metric_name: tp.Optional[str] = 'ce'

        self._cached_batch_writer = None
        self._cached_batch_loader = None
        if cfg.cache.path:
            if cfg.cache.write:
                self._cached_batch_writer = CachedBatchWriter(Path(cfg.cache.path))
                if self.cfg.cache.write_num_shards:
                    self.logger.warning("Multiple shard cache, best_metric_name will be set to None.")
                    self._best_metric_name = None
            else:
                self._cached_batch_loader = CachedBatchLoader(
                    Path(cfg.cache.path), cfg.dataset.batch_size, cfg.dataset.num_workers,
                    min_length=self.cfg.optim.updates_per_epoch or 1)
                self.dataloaders['original_train'] = self.dataloaders['train']
                self.dataloaders['train'] = self._cached_batch_loader  # type: ignore
        # import demucs to source sep the music
        self.__dict__['demucs'] = pretrained.get_model('htdemucs').to(self.device)
        demucs_sources: list = self.demucs.sources  # type: ignore
        sources = self.cfg.multistem_compression_model_checkpoints.sources
        self.stem_indices = torch.LongTensor([demucs_sources.index(source) for source in sources]).to(self.device)
        self.num_codebooks_per_stem = self.multistem_compression_model.num_codebooks

    @staticmethod
    def get_eval_solver_from_sig(sig: str, dtype: tp.Optional[str] = None,
                                 device: tp.Optional[str] = None, autocast: bool = True,
                                 batch_size: tp.Optional[int] = None,
                                 override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
                                 **kwargs):
        """Mostly a convenience function around magma.train.get_solver_from_sig,
        populating all the proper param, deactivating EMA, FSDP, loading the best state,
        basically all you need to get a solver ready to "play" with in single GPU mode
        and with minimal memory overhead.

        Args:
            sig (str): signature to load.
            dtype (str or None): potential dtype, as a string, i.e. 'float16'.
            device (str or None): potential device, as a string, i.e. 'cuda'.
            override_cfg (dict or omegaconf.DictConfig or None): potential device, as a string, i.e. 'cuda'.
        """
        from audiocraft import train
        our_override_cfg: tp.Dict[str, tp.Any] = {'optim': {'ema': {'use': False}}}
        our_override_cfg['autocast'] = autocast
        if dtype is not None:
            our_override_cfg['dtype'] = dtype
        if device is not None:
            our_override_cfg['device'] = device
        if batch_size is not None:
            our_override_cfg['dataset'] = {'batch_size': batch_size}
        if override_cfg is None:
            override_cfg = {}
        override_cfg = omegaconf.OmegaConf.merge(
            omegaconf.DictConfig(override_cfg), omegaconf.DictConfig(our_override_cfg))  # type: ignore
        solver = train.get_solver_from_sig(
            sig, override_cfg=override_cfg,
            load_best=True, disable_fsdp=True,
            ignore_state_keys=['optimizer', 'ema'], **kwargs)
        solver.model.eval()
        return solver

    def get_formatter(self, stage_name: str) -> flashy.Formatter:
        return flashy.Formatter({
            'lr': '.2E',
            'ce': '.3f',
            'ppl': '.3f',
            'grad_norm': '.3E',
        }, exclude_keys=['ce_q*', 'ppl_q*'])

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        return self._best_metric_name

    def build_model(self) -> None:
        """Instantiate models and optimizer."""
        # we can potentially not use all quantizers with which the EnCodec model was trained
        # (e.g. we trained the model with quantizers dropout)
        if self.cfg.multistem_compression_model_checkpoints.pretrained is not None:
            self.multistem_compression_model = models.MultiStemCompressionModel.get_pretrained(
                    self.cfg.multistem_compression_model_checkpoints.pretrained, device=self.device)
        else:
            sources = self.cfg.multistem_compression_model_checkpoints.sources
            all_compression_models = [CompressionSolver.model_from_checkpoint(self.cfg.multistem_compression_model_checkpoints[source]).to(self.device) for source in sources]
            self.multistem_compression_model = models.MultiStemCompressionModel(sources, all_compression_models).to(self.device)

        assert self.multistem_compression_model.sample_rate == self.cfg.sample_rate, (
            f"Compression model sample rate is {self.multistem_compression_model.sample_rate} but "
            f"Solver sample rate is {self.cfg.sample_rate}."
            )
        # ensure we have matching configuration between LM and compression model
        assert self.cfg.transformer_lm.card == self.multistem_compression_model.cardinality['bass'], (
            "Cardinalities of the LM and compression model don't match: ",
            f"LM cardinality is {self.cfg.transformer_lm.card} vs ",
            f"compression model cardinality is {self.multistem_compression_model.cardinality}"
        )
        assert self.cfg.transformer_lm.n_q == sum(self.multistem_compression_model.num_codebooks.values()), (
            "Numbers of codebooks of the LM and compression models don't match: ",
            f"LM number of codebooks is {self.cfg.transformer_lm.n_q} vs ",
            f"compression model numer of codebooks is {sum(self.multistem_compression_model.num_codebooks.values())}"
        )
        self.logger.info("Compression model has these codebooks: %s, this cardinality: %s, and a framerate of %d",
                         self.multistem_compression_model.num_codebooks, self.multistem_compression_model.cardinality,
                         self.multistem_compression_model.frame_rate)
        # instantiate LM model
        self.model: models.LMModel = models.builders.get_lm_model(self.cfg).to(self.device)
        if self.cfg.fsdp.use:
            assert not self.cfg.autocast, "Cannot use autocast with fsdp"
            self.model = self.wrap_with_fsdp(self.model)
        self.register_ema('model')
        # initialize optimization
        self.optimizer = builders.get_optimizer(builders.get_optim_parameter_groups(self.model), self.cfg.optim)
        self.lr_scheduler = builders.get_lr_scheduler(self.optimizer, self.cfg.schedule, self.total_updates)
        self.register_stateful('model', 'optimizer', 'lr_scheduler')
        self.register_best_state('model')
        self.autocast_dtype = {
            'float16': torch.float16, 'bfloat16': torch.bfloat16
        }[self.cfg.autocast_dtype]
        self.scaler: tp.Optional[torch.cuda.amp.GradScaler] = None
        if self.cfg.fsdp.use:
            need_scaler = self.cfg.fsdp.param_dtype == 'float16'
        else:
            need_scaler = self.cfg.autocast and self.autocast_dtype is torch.float16
        if need_scaler:
            if self.cfg.fsdp.use:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
                self.scaler = ShardedGradScaler()  # type: ignore
            else:
                self.scaler = torch.cuda.amp.GradScaler()
            self.register_stateful('scaler')

    def build_dataloaders(self) -> None:
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg, dataset_type=self.DATASET_TYPE)

    def show(self) -> None:
        """Show the compression model and LM model."""
        self.logger.info("Compression model:")
        self.log_model_summary(self.multistem_compression_model)
        self.logger.info("LM model:")
        self.log_model_summary(self.model)

    def load_state_dict(self, state: dict) -> None:
        if 'condition_provider' in state:
            model_state = state['model']
            condition_provider_state = state.pop('condition_provider')
            prefix = 'condition_provider.'
            for key, value in condition_provider_state.items():
                key = prefix + key
                assert key not in model_state
                model_state[key] = value
        if 'compression_model' in state:
            # We used to store the `compression_model` state in the checkpoint, however
            # this is in general not needed, as the compression model should always be readable
            # from the original `cfg.compression_model_checkpoint` location.
            compression_model_state = state.pop('compression_model')
            before_hash = model_hash(self.multistem_compression_model)
            self.multistem_compression_model.load_state_dict(compression_model_state)
            after_hash = model_hash(self.multistem_compression_model)
            if before_hash != after_hash:
                raise RuntimeError(
                    "The compression model state inside the checkpoint is different"
                    " from the one obtained from compression_model_checkpoint..."
                    "We do not support altering the compression model inside the LM "
                    "checkpoint as parts of the code, in particular for running eval post-training "
                    "will use the compression_model_checkpoint as the source of truth.")

        super().load_state_dict(state)

    def load_from_pretrained(self, name: str):
        # TODO: support native HF versions of MusicGen.
        lm_pkg = models.loaders.load_lm_model_ckpt(name)
        state: dict = {
            'best_state': {
                'model': lm_pkg['best_state'],
            },
        }
        return state

    @torch.no_grad()
    def _compute_mixture_to_stems(self, wav: torch.Tensor) -> torch.Tensor:
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        with self.autocast:
            wav = wav.to(self.device)
            wav = convert_audio(
                wav, self.cfg.sample_rate, self.demucs.samplerate, self.demucs.audio_channels)  # type: ignore
            stems = apply_model(self.demucs, wav, device=self.device)
            stems = stems[:, self.stem_indices]  # extract relevant stems for melody conditioning
            stems = convert_audio(stems, self.demucs.samplerate, self.cfg.sample_rate, 1)  # type: ignore
            return stems

    def _compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def _select_instruments_to_mask(self, B, mask_other_fine_tokens=False) -> tp.List[int]:
        '''
        it uses self.num_codebooks_per_stem 
            for instance if self.num_codebooks_per_stem= {'bass': 2, 'drums': 2, 'other': 3}
        it selects between 1 and num_instruments - 1 to mask and returns their corresponding
        indices in the codebook streams
        for instance if it selects 'bass' and 'other' it returns [0, 1, 4, 5, 6]  
        '''

        # Create a list of all instrument indices
        instrument_indices = []
        start_index = 0
        for instrument, num_codebooks in self.num_codebooks_per_stem.items():
            indices = list(range(start_index, start_index + num_codebooks))
            instrument_indices.append(indices)
            start_index += num_codebooks
        
        # List of instruments
        instruments = list(self.num_codebooks_per_stem.keys())
        num_instruments = len(instruments)
        
        # Result list of lists for B iterations
        result_masked_indices = []
        # Iterate B times
        for _ in range(B):
            # Randomly decide how many instruments to mask (at least 1, at most N-1)
            num_to_mask = random.randint(1, num_instruments - 1)
            
            # Randomly select instruments to mask
            masked_instruments = random.sample(instruments, num_to_mask)
            
            # Compile indices of the masked instruments
            masked_indices = []
            for instrument in masked_instruments:
                index = instruments.index(instrument)
                masked_indices.extend(instrument_indices[index])
            if mask_other_fine_tokens:
                # we assume other is the last key in the dictionary
                masked_indices.extend(instrument_indices[-1][1:])
                masked_indices = list(set(masked_indices))
            
            # Append the result for this iteration
            result_masked_indices.append(masked_indices)
        return result_masked_indices

    def _create_downsampled_prefix_with_masked_instruments(self, tokens: torch.Tensor, B_idx_to_mask: tp.List[tp.List]):
        B, K, T = tokens.shape
        assert B == len(B_idx_to_mask)
        ds = self.cfg.instrument_masking.ds_factor
        T_new = int(ds / (ds + 1) * T)
        token_unmask = tokens.clone()
        token_unmask = token_unmask[..., :T_new]
        token_ds_mask = token_unmask.clone()[..., ::ds]

        for i in range(B):
            # Mask tokens from 0 to t with special_token_id
            token_ds_mask[i, B_idx_to_mask[i]] = self.model.special_token_id
        transition_token = self.model.special_token_id * torch.ones((B, K, 1), device=tokens.device, dtype=tokens.dtype)
        output_token = torch.cat([token_ds_mask, transition_token, token_unmask], dim=-1)

        ce_mask = torch.zeros_like(output_token, dtype=torch.bool, device=output_token.device)
        ce_mask[..., -T_new:] = 1

        return output_token, ce_mask
    

    def _prepare_tokens_and_attributes(
        self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
        check_synchronization_points: bool = False
    ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        if self.model.training:
            warnings.warn(
                "Up to version 1.0.1, the _prepare_tokens_and_attributes was evaluated with `torch.no_grad()`. "
                "This is inconsistent with how model were trained in the MusicGen paper. We removed the "
                "`torch.no_grad()` in version 1.1.0. Small changes to the final performance are expected. "
                "Really sorry about that.")
        if self._cached_batch_loader is None or self.current_stage != "train":
            audio, infos = batch
            audio = audio.to(self.device)
            audio_tokens = None
            assert audio.size(0) == len(infos), (
                f"Mismatch between number of items in audio batch ({audio.size(0)})",
                f" and in metadata ({len(infos)})"
            )
        else:
            audio = None
            # In that case the batch will be a tuple coming from the _cached_batch_writer bit below.
            infos, = batch  # type: ignore
            assert all([isinstance(info, AudioInfo) for info in infos])
            assert all([info.audio_tokens is not None for info in infos])  # type: ignore
            audio_tokens = torch.stack([info.audio_tokens for info in infos]).to(self.device)  # type: ignore
            audio_tokens = audio_tokens.long()
            for info in infos:
                if isinstance(info, MusicInfo):
                    # Careful here, if you want to use this condition_wav (e.b. chroma conditioning),
                    # then you must be using the chroma cache! otherwise the code will try
                    # to use this segment and fail (by that I mean you will see NaN everywhere).
                    info.self_wav = WavCondition(
                        torch.full([1, info.channels, info.total_frames], float('NaN')),
                        length=torch.tensor([info.n_frames]),
                        sample_rate=[info.sample_rate],
                        path=[info.meta.path],
                        seek_time=[info.seek_time])
                    dataset = get_dataset_from_loader(self.dataloaders['original_train'])
                    assert isinstance(dataset, MusicDataset), type(dataset)
                    if dataset.paraphraser is not None and info.description is not None:
                        # Hackingly reapplying paraphraser when using cache.
                        info.description = dataset.paraphraser.sample_paraphrase(
                            info.meta.path, info.description)
        # prepare attributes
        attributes = [info.to_condition_attributes() for info in infos]
        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        # Now we should be synchronization free.
        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        if audio_tokens is None:
            with torch.no_grad():
                # we assume audio is [B, num_stems, C, T] C=1 or 2
                audio_tokens = self.multistem_compression_model.encode(audio) # [B, num_streams, T]

        ce_mask = None
        if self.cfg.instrument_masking_prob > 0:
            if random.random() < self.cfg.instrument_masking_prob:
                B_idx_to_mask = self._select_instruments_to_mask(audio_tokens.shape[0])
                audio_tokens, ce_mask = self._create_downsampled_prefix_with_masked_instruments(audio_tokens, B_idx_to_mask)

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
        # replace encodec tokens from padded audio with special_token_id
        if self.cfg.tokens.padding_with_special_token:
            audio_tokens = audio_tokens.clone()
            padding_mask = padding_mask.clone()
            token_sample_rate = self.multistem_compression_model.frame_rate
            B, K, T_s = audio_tokens.shape
            for i in range(B):
                n_samples = infos[i].n_frames
                audio_sample_rate = infos[i].sample_rate
                # take the last token generated from actual audio frames (non-padded audio)
                valid_tokens = math.floor(float(n_samples) / audio_sample_rate * token_sample_rate)
                audio_tokens[i, :, valid_tokens:] = self.model.special_token_id
                padding_mask[i, :, valid_tokens:] = 0

        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("default")

        if self._cached_batch_writer is not None and self.current_stage == 'train':
            assert self._cached_batch_loader is None
            assert audio_tokens is not None
            for info, one_audio_tokens in zip(infos, audio_tokens):
                assert isinstance(info, AudioInfo)
                if isinstance(info, MusicInfo):
                    assert not info.joint_embed, "joint_embed and cache not supported yet."
                    info.self_wav = None
                assert one_audio_tokens.max() < 2**15, one_audio_tokens.max().item()
                info.audio_tokens = one_audio_tokens.short().cpu()
            self._cached_batch_writer.save(infos)

        if ce_mask is not None:
            padding_mask = padding_mask & ce_mask
        return condition_tensors, audio_tokens, padding_mask

    def run_step(self, idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict:
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == 'cuda'

        mixture, info = batch
        audio = self._compute_mixture_to_stems(mixture)
        if hasattr(self.model.condition_provider.conditioners, 'self_wav'):
            if isinstance(self.model.condition_provider.conditioners.self_wav, MultiStemStyleConditioner):
                for i, elem in enumerate(info):
                    if elem.self_wav.wav.shape[-1] > 1: 
                        elem.self_wav = WavCondition(wav=audio[i, :, 0][None], 
                                            length=elem.self_wav.length,
                                            sample_rate=elem.self_wav.sample_rate,
                                            path=elem.self_wav.path,
                                            seek_time=elem.self_wav.seek_time,
                                            )

        batch = (audio, info)
        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
            batch, check_synchronization_points)

        self.deadlock_detect.update('tokens_and_conditions')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('warn')

        with self.autocast:
            style_mask = None
            if hasattr(self.model.condition_provider.conditioners, 'self_wav'):
                if isinstance(self.model.condition_provider.conditioners.self_wav, MultiStemStyleConditioner):
                    style_mask = self.model.condition_provider.conditioners.self_wav.mask

            model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
            logits = model_output.logits
            if style_mask is not None:
                mask = padding_mask & model_output.mask & style_mask
            else:
                mask = padding_mask & model_output.mask
            ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
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
        for k, ce_q in enumerate(ce_per_codebook):
            metrics[f'ce_q{k + 1}'] = ce_q
            metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)

        return metrics

    @torch.no_grad()
    def run_generate_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
                          gen_duration: float, prompt_duration: tp.Optional[float] = None,
                          remove_prompt: bool = False, remove_text_conditioning: bool = False,
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
        mixture, meta = batch
        audio = self._compute_mixture_to_stems(mixture)
        if hasattr(self.model.condition_provider.conditioners, 'self_wav'):
            if isinstance(self.model.condition_provider.conditioners.self_wav, MultiStemStyleConditioner):
                for i, elem in enumerate(meta):
                    if elem.self_wav.wav.shape[-1] > 1: 
                        elem.self_wav = WavCondition(wav=audio[i, :, 0][None].to(self.device), 
                                            length=elem.self_wav.length.to(self.device),
                                            sample_rate=elem.self_wav.sample_rate,
                                            path=elem.self_wav.path,
                                            seek_time=elem.self_wav.seek_time,
                                            )

        batch = (audio, meta)

        assert audio.size(0) == len(meta), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(meta)})"
        )
        # prepare attributes
        attributes = [x.to_condition_attributes() for x in meta]
        # TODO: Add dropout for chroma?
        if remove_text_conditioning:
            attributes = AttributeDropout(p={'text':{'description': 1.0}})(attributes)

        # prepare audio prompt
        if prompt_duration is None:
            prompt_audio = None
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_audio_frames = int(prompt_duration * self.multistem_compression_model.sample_rate)
            prompt_audio = audio[..., :prompt_audio_frames]

        # get audio tokens from compression model
        if prompt_audio is None or prompt_audio.nelement() == 0:
            num_samples = len(attributes)
            prompt_tokens = None
        else:
            num_samples = None
            prompt_audio = prompt_audio.to(self.device)
            prompt_tokens = self.multistem_compression_model.encode(prompt_audio)

        # generate by sampling from the LM
        with self.autocast:
            total_gen_len = math.ceil(gen_duration * self.multistem_compression_model.frame_rate)
            gen_tokens = self.model.generate(
                prompt_tokens, attributes, max_gen_len=total_gen_len,
                num_samples=num_samples, **self.generation_params)

        # generate audio from tokens
        assert gen_tokens.dim() == 3
        gen_audio = self.multistem_compression_model.decode(gen_tokens) # dict
        gen_audio['gen_audio'] = sum(gen_audio.values())

        bench_end = time.time()
        gen_outputs = {
            'rtf': (bench_end - bench_start) / gen_duration,
            'ref_audio': audio,
            'gen_tokens': gen_tokens,
            'prompt_audio': prompt_audio.sum(1) if prompt_audio is not None else None,
            'prompt_tokens': prompt_tokens,
        }
        gen_outputs = {**gen_outputs, **gen_audio}
        return gen_outputs

    def generate_audio(self) -> dict:
        """Audio generation stage."""
        generate_stage_name = f'{self.current_stage}'
        sample_manager = SampleManager(self.xp)
        self.logger.info(f"Generating samples in {sample_manager.base_folder}")
        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        dataset = get_dataset_from_loader(loader)
        dataset_duration = dataset.segment_duration
        assert dataset_duration is not None
        assert isinstance(dataset, AudioDataset)
        target_duration = self.cfg.generate.lm.gen_duration
        prompt_duration = self.cfg.generate.lm.prompt_duration
        if target_duration is None:
            target_duration = dataset_duration
        if prompt_duration is None:
            prompt_duration = dataset_duration / 4
        assert prompt_duration < dataset_duration, (
            f"Specified prompt duration ({prompt_duration}s) is longer",
            f" than reference audio duration ({dataset_duration}s)"
        )

        def get_hydrated_conditions(meta: tp.List[SegmentWithAttributes]):
            hydrated_conditions = []
            for sample in [x.to_condition_attributes() for x in meta]:
                cond_dict = {}
                for cond_type in sample.__annotations__.keys():
                    for cond_key, cond_val in getattr(sample, cond_type).items():
                        if cond_key not in self.model.condition_provider.conditioners.keys():
                            continue
                        if is_jsonable(cond_val):
                            cond_dict[cond_key] = cond_val
                        elif isinstance(cond_val, WavCondition):
                            cond_dict[cond_key] = cond_val.path
                        elif isinstance(cond_val, JointEmbedCondition):
                            cond_dict[cond_key] = cond_val.text  # only support text at inference for now
                        else:
                            # if we reached this point, it is not clear how to log the condition
                            # so we just log the type.
                            cond_dict[cond_key] = str(type(cond_val))
                            continue
                hydrated_conditions.append(cond_dict)
            return hydrated_conditions

        metrics: dict = {}
        average = flashy.averager()
        for batch in lp:
            audio, meta = batch
            # metadata for sample manager
            hydrated_conditions = get_hydrated_conditions(meta)
            sample_generation_params = {
                **{f'classifier_free_guidance_{k}': v for k, v in self.cfg.classifier_free_guidance.items()},
                **self.generation_params
            }
            if self.cfg.generate.lm.unprompted_samples:
                if self.cfg.generate.lm.gen_gt_samples:
                    # get the ground truth instead of generation
                    self.logger.warn(
                        "Use ground truth instead of audio generation as generate.lm.gen_gt_samples=true")
                    gen_unprompted_audio = audio
                    rtf = 1.
                else:
                    gen_unprompted_outputs = self.run_generate_step(
                        batch, gen_duration=target_duration, prompt_duration=None,
                        **self.generation_params)
                    gen_unprompted_audio = gen_unprompted_outputs['gen_audio'].cpu()
                    rtf = gen_unprompted_outputs['rtf']
                sample_manager.add_samples(
                    gen_unprompted_audio, self.epoch, hydrated_conditions,
                    ground_truth_wavs=audio, generation_args=sample_generation_params)

            if self.cfg.generate.lm.prompted_samples:
                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration, prompt_duration=prompt_duration,
                    **self.generation_params)
                gen_audio = gen_outputs['gen_audio'].cpu()
                prompt_audio = gen_outputs['prompt_audio'].cpu()
                sample_manager.add_samples(
                    gen_audio, self.epoch, hydrated_conditions,
                    prompt_wavs=prompt_audio, ground_truth_wavs=audio,
                    generation_args=sample_generation_params)

            metrics['rtf'] = rtf
            metrics = average(metrics)

        flashy.distrib.barrier()
        return metrics

    def generate(self) -> dict:
        """Generate stage."""
        self.model.eval()
        with torch.no_grad():
            return self.generate_audio()

    def run_epoch(self):
        if self.cfg.cache.write:
            if ((self.epoch - 1) % self.cfg.cache.write_num_shards) != self.cfg.cache.write_shard:
                return
        super().run_epoch()

    def train(self):
        """Train stage.
        """
        if self._cached_batch_writer is not None:
            self._cached_batch_writer.start_epoch(self.epoch)
        if self._cached_batch_loader is None:
            dataset = get_dataset_from_loader(self.dataloaders['train'])
            assert isinstance(dataset, AudioDataset)
            dataset.current_epoch = self.epoch
        else:
            self._cached_batch_loader.start_epoch(self.epoch)
        return super().train()

    def evaluate_audio_generation(self) -> dict:
        """Evaluate audio generation with off-the-shelf metrics."""
        evaluate_stage_name = f'{self.current_stage}_generation'
        # instantiate evaluation metrics, if at least one metric is defined, run audio generation evaluation
        fad: tp.Optional[eval_metrics.FrechetAudioDistanceMetric] = None
        kldiv: tp.Optional[eval_metrics.KLDivergenceMetric] = None
        text_consistency: tp.Optional[eval_metrics.TextConsistencyMetric] = None
        chroma_cosine: tp.Optional[eval_metrics.ChromaCosineSimilarityMetric] = None
        should_run_eval = False
        eval_chroma_wavs: tp.Optional[torch.Tensor] = None
        if self.cfg.evaluate.metrics.fad:
            fad = builders.get_fad(self.cfg.metrics.fad).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.kld:
            kldiv = builders.get_kldiv(self.cfg.metrics.kld).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.text_consistency:
            text_consistency = builders.get_text_consistency(self.cfg.metrics.text_consistency).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.chroma_cosine:
            chroma_cosine = builders.get_chroma_cosine_similarity(self.cfg.metrics.chroma_cosine).to(self.device)
            # if we have predefind wavs for chroma we should purge them for computing the cosine metric
            has_predefined_eval_chromas = 'self_wav' in self.model.condition_provider.conditioners and \
                                          self.model.condition_provider.conditioners['self_wav'].has_eval_wavs()
            if has_predefined_eval_chromas:
                warn_once(self.logger, "Attempting to run cosine eval for config with pre-defined eval chromas! "
                                       'Resetting eval chromas to None for evaluation.')
                eval_chroma_wavs = self.model.condition_provider.conditioners.self_wav.eval_wavs  # type: ignore
                self.model.condition_provider.conditioners.self_wav.reset_eval_wavs(None)  # type: ignore
            should_run_eval = True

        def get_compressed_audio(audio: torch.Tensor) -> torch.Tensor:
            audio_tokens = self.multistem_compression_model.encode(audio.to(self.device))
            compressed_audio = self.multistem_compression_model.decode(audio_tokens)
            mixture = sum(compressed_audio.values())
            return mixture[..., :audio.shape[-1]]

        metrics: dict = {}
        if should_run_eval:
            loader = self.dataloaders['evaluate']
            updates = len(loader)
            lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
            average = flashy.averager()
            dataset = get_dataset_from_loader(loader)
            assert isinstance(dataset, AudioDataset)
            self.logger.info(f"Computing evaluation metrics on {len(dataset)} samples")

            for idx, batch in enumerate(lp):
                audio, meta = batch
                assert all([self.cfg.sample_rate == m.sample_rate for m in meta])

                target_duration = audio.shape[-1] / self.cfg.sample_rate
                if self.cfg.evaluate.fixed_generation_duration:
                    target_duration = self.cfg.evaluate.fixed_generation_duration
                if self.cfg.evaluate.remove_text_conditioning:
                    remove_text_conditioning = self.cfg.evaluate.remove_text_conditioning
                else:
                    remove_text_conditioning = False

                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration, remove_text_conditioning=remove_text_conditioning,
                    **self.generation_params
                )
                y_pred = gen_outputs['gen_audio'].detach()
                y_pred = y_pred[..., :audio.shape[-1]]

                normalize_kwargs = dict(self.cfg.generate.audio)
                normalize_kwargs.pop('format', None)
                y_pred = torch.stack([normalize_audio(w, **normalize_kwargs) for w in y_pred], dim=0).cpu()
                y = audio.cpu()  # should already be on CPU but just in case
                sizes = torch.tensor([m.n_frames for m in meta])  # actual sizes without padding
                sample_rates = torch.tensor([m.sample_rate for m in meta])  # sample rates for audio samples
                audio_stems = [Path(m.meta.path).stem + f"_{m.seek_time}" for m in meta]

                if fad is not None:
                    if self.cfg.metrics.fad.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    fad.update(y_pred, y, sizes, sample_rates, audio_stems)
                if kldiv is not None:
                    if self.cfg.metrics.kld.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    kldiv.update(y_pred, y, sizes, sample_rates)
                if text_consistency is not None:
                    texts = [m.description for m in meta]
                    if self.cfg.metrics.text_consistency.use_gt:
                        y_pred = y
                    text_consistency.update(y_pred, texts, sizes, sample_rates)
                if chroma_cosine is not None:
                    if self.cfg.metrics.chroma_cosine.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    chroma_cosine.update(y_pred, y, sizes, sample_rates)
                    # restore chroma conditioner's eval chroma wavs
                    if eval_chroma_wavs is not None:
                        self.model.condition_provider.conditioners['self_wav'].reset_eval_wavs(eval_chroma_wavs)

            flashy.distrib.barrier()
            if fad is not None:
                metrics['fad'] = fad.compute()
            if kldiv is not None:
                kld_metrics = kldiv.compute()
                metrics.update(kld_metrics)
            if text_consistency is not None:
                metrics['text_consistency'] = text_consistency.compute()
            if chroma_cosine is not None:
                metrics['chroma_cosine'] = chroma_cosine.compute()
            metrics = average(metrics)
            metrics = flashy.distrib.average_metrics(metrics, len(loader))

        return metrics

    def evaluate(self) -> dict:
        """Evaluate stage."""
        self.model.eval()
        with torch.no_grad():
            metrics: dict = {}
            if self.cfg.evaluate.metrics.base:
                metrics.update(self.common_train_valid('evaluate'))
            gen_metrics = self.evaluate_audio_generation()
            return {**metrics, **gen_metrics}
