# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import flashy
import julius
import omegaconf
import torch
import torch.nn.functional as F

from . import builders
from . import base
from .. import models
from ..modules.diffusion_schedule import NoiseSchedule
from ..metrics import RelativeVolumeMel
from ..models.builders import get_processor
from ..utils.samples.manager import SampleManager
from ..solvers.compression import CompressionSolver


class PerStageMetrics:
    """Handle prompting the metrics per stage.
    It outputs the metrics per range of diffusion states.
    e.g. avg loss when t in [250, 500]
    """
    def __init__(self, num_steps: int, num_stages: int = 4):
        self.num_steps = num_steps
        self.num_stages = num_stages

    def __call__(self, losses: dict, step: tp.Union[int, torch.Tensor]):
        if type(step) is int:
            stage = int((step / self.num_steps) * self.num_stages)
            return {f"{name}_{stage}": loss for name, loss in losses.items()}
        elif type(step) is torch.Tensor:
            stage_tensor = ((step / self.num_steps) * self.num_stages).long()
            out: tp.Dict[str, float] = {}
            for stage_idx in range(self.num_stages):
                mask = (stage_tensor == stage_idx)
                N = mask.sum()
                stage_out = {}
                if N > 0:  # pass if no elements in the stage
                    for name, loss in losses.items():
                        stage_loss = (mask * loss).sum() / N
                        stage_out[f"{name}_{stage_idx}"] = stage_loss
                out = {**out, **stage_out}
            return out


class DataProcess:
    """Apply filtering or resampling.

    Args:
        initial_sr (int): Initial sample rate.
        target_sr (int): Target sample rate.
        use_resampling: Whether to use resampling or not.
        use_filter (bool):
        n_bands (int): Number of bands to consider.
        idx_band (int):
        device (torch.device or str):
        cutoffs ():
        boost (bool):
    """
    def __init__(self, initial_sr: int = 24000, target_sr: int = 16000, use_resampling: bool = False,
                 use_filter: bool = False, n_bands: int = 4,
                 idx_band: int = 0, device: torch.device = torch.device('cpu'), cutoffs=None, boost=False):
        """Apply filtering or resampling
        Args:
            initial_sr (int): sample rate of the dataset
            target_sr (int): sample rate after resampling
            use_resampling (bool): whether or not performs resampling
            use_filter (bool): when True filter the data to keep only one frequency band
            n_bands (int): Number of bands used
            cuts (none or list): The cutoff frequencies of the band filtering
                                if None then we use mel scale bands.
            idx_band (int): index of the frequency band. 0 are lows ... (n_bands - 1) highs
            boost (bool): make the data scale match our music dataset.
        """
        assert idx_band < n_bands
        self.idx_band = idx_band
        if use_filter:
            if cutoffs is not None:
                self.filter = julius.SplitBands(sample_rate=initial_sr, cutoffs=cutoffs).to(device)
            else:
                self.filter = julius.SplitBands(sample_rate=initial_sr, n_bands=n_bands).to(device)
        self.use_filter = use_filter
        self.use_resampling = use_resampling
        self.target_sr = target_sr
        self.initial_sr = initial_sr
        self.boost = boost

    def process_data(self, x, metric=False):
        if x is None:
            return None
        if self.boost:
            x /= torch.clamp(x.std(dim=(1, 2), keepdim=True), min=1e-4)
            x * 0.22
        if self.use_filter and not metric:
            x = self.filter(x)[self.idx_band]
        if self.use_resampling:
            x = julius.resample_frac(x, old_sr=self.initial_sr, new_sr=self.target_sr)
        return x

    def inverse_process(self, x):
        """Upsampling only."""
        if self.use_resampling:
            x = julius.resample_frac(x, old_sr=self.target_sr, new_sr=self.target_sr)
        return x


class DiffusionSolver(base.StandardSolver):
    """Solver for compression task.

    The diffusion task allows for MultiBand diffusion model training.

    Args:
        cfg (DictConfig): Configuration.
    """
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = cfg.device
        self.sample_rate: int = self.cfg.sample_rate
        self.codec_model = CompressionSolver.model_from_checkpoint(
            cfg.compression_model_checkpoint, device=self.device)

        self.codec_model.set_num_codebooks(cfg.n_q)
        assert self.codec_model.sample_rate == self.cfg.sample_rate, (
            f"Codec model sample rate is {self.codec_model.sample_rate} but "
            f"Solver sample rate is {self.cfg.sample_rate}."
            )
        assert self.codec_model.sample_rate == self.sample_rate, \
            f"Sample rate of solver {self.sample_rate} and codec {self.codec_model.sample_rate} " \
            "don't match."

        self.sample_processor = get_processor(cfg.processor, sample_rate=self.sample_rate)
        self.register_stateful('sample_processor')
        self.sample_processor.to(self.device)

        self.schedule = NoiseSchedule(
            **cfg.schedule, device=self.device, sample_processor=self.sample_processor)

        self.eval_metric: tp.Optional[torch.nn.Module] = None

        self.rvm = RelativeVolumeMel()
        self.data_processor = DataProcess(initial_sr=self.sample_rate, target_sr=cfg.resampling.target_sr,
                                          use_resampling=cfg.resampling.use, cutoffs=cfg.filter.cutoffs,
                                          use_filter=cfg.filter.use, n_bands=cfg.filter.n_bands,
                                          idx_band=cfg.filter.idx_band, device=self.device)

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        if self._current_stage == "evaluate":
            return 'rvm'
        else:
            return 'loss'

    @torch.no_grad()
    def get_condition(self, wav: torch.Tensor) -> torch.Tensor:
        codes, scale = self.codec_model.encode(wav)
        assert scale is None, "Scaled compression models not supported."
        emb = self.codec_model.decode_latent(codes)
        return emb

    def build_model(self):
        """Build model and optimizer as well as optional Exponential Moving Average of the model.
        """
        # Model and optimizer
        self.model = models.builders.get_diffusion_model(self.cfg).to(self.device)
        self.optimizer = builders.get_optimizer(self.model.parameters(), self.cfg.optim)
        self.register_stateful('model', 'optimizer')
        self.register_best_state('model')
        self.register_ema('model')

    def build_dataloaders(self):
        """Build audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg)

    def show(self):
        # TODO
        raise NotImplementedError()

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        x = batch.to(self.device)
        loss_fun = F.mse_loss if self.cfg.loss.kind == 'mse' else F.l1_loss

        condition = self.get_condition(x)  # [bs, 128, T/hop, n_emb]
        sample = self.data_processor.process_data(x)

        input_, target, step = self.schedule.get_training_item(sample,
                                                               tensor_step=self.cfg.schedule.variable_step_batch)
        out = self.model(input_, step, condition=condition).sample

        base_loss = loss_fun(out, target, reduction='none').mean(dim=(1, 2))
        reference_loss = loss_fun(input_, target, reduction='none').mean(dim=(1, 2))
        loss = base_loss / reference_loss ** self.cfg.loss.norm_power

        if self.is_training:
            loss.mean().backward()
            flashy.distrib.sync_model(self.model)
            self.optimizer.step()
            self.optimizer.zero_grad()
        metrics = {
            'loss': loss.mean(), 'normed_loss': (base_loss / reference_loss).mean(),
            }
        metrics.update(self.per_stage({'loss': loss, 'normed_loss': base_loss / reference_loss}, step))
        metrics.update({
            'std_in': input_.std(), 'std_out': out.std()})
        return metrics

    def run_epoch(self):
        # reset random seed at the beginning of the epoch
        self.rng = torch.Generator()
        self.rng.manual_seed(1234 + self.epoch)
        self.per_stage = PerStageMetrics(self.schedule.num_steps, self.cfg.metrics.num_stage)
        # run epoch
        super().run_epoch()

    def evaluate(self):
        """Evaluate stage.
        Runs audio reconstruction evaluation.
        """
        self.model.eval()
        evaluate_stage_name = f'{self.current_stage}'
        loader = self.dataloaders['evaluate']
        updates = len(loader)
        lp = self.log_progress(f'{evaluate_stage_name} estimate', loader, total=updates, updates=self.log_updates)

        metrics = {}
        n = 1
        for idx, batch in enumerate(lp):
            x = batch.to(self.device)
            with torch.no_grad():
                y_pred = self.regenerate(x)

            y_pred = y_pred.cpu()
            y = batch.cpu()  # should already be on CPU but just in case
            rvm = self.rvm(y_pred, y)
            lp.update(**rvm)
            if len(metrics) == 0:
                metrics = rvm
            else:
                for key in rvm.keys():
                    metrics[key] = (metrics[key] * n + rvm[key]) / (n + 1)
        metrics = flashy.distrib.average_metrics(metrics)
        return metrics

    @torch.no_grad()
    def regenerate(self, wav: torch.Tensor, step_list: tp.Optional[list] = None):
        """Regenerate the given waveform."""
        condition = self.get_condition(wav)
        initial = self.schedule.get_initial_noise(self.data_processor.process_data(wav))  # sampling rate changes.
        result = self.schedule.generate_subsampled(self.model, initial=initial, condition=condition,
                                                   step_list=step_list)
        result = self.data_processor.inverse_process(result)
        return result

    def generate(self):
        """Generate stage."""
        sample_manager = SampleManager(self.xp)
        self.model.eval()
        generate_stage_name = f'{self.current_stage}'

        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        for batch in lp:
            reference, _ = batch
            reference = reference.to(self.device)
            estimate = self.regenerate(reference)
            reference = reference.cpu()
            estimate = estimate.cpu()
            sample_manager.add_samples(estimate, self.epoch, ground_truth_wavs=reference)
        flashy.distrib.barrier()
