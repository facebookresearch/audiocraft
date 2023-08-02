# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import multiprocessing
from pathlib import Path
import typing as tp

import flashy
import omegaconf
import torch
from torch import nn

from . import base, builders
from .. import models, quantization
from ..utils import checkpoint
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_pool_executor


logger = logging.getLogger(__name__)


class CompressionSolver(base.StandardSolver):
    """Solver for compression task.

    The compression task combines a set of perceptual and objective losses
    to train an EncodecModel (composed of an encoder-decoder and a quantizer)
    to perform high fidelity audio reconstruction.
    """
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.rng: torch.Generator  # set at each epoch
        self.adv_losses = builders.get_adversarial_losses(self.cfg)
        self.aux_losses = nn.ModuleDict()
        self.info_losses = nn.ModuleDict()
        assert not cfg.fsdp.use, "FSDP not supported by CompressionSolver."
        loss_weights = dict()
        for loss_name, weight in self.cfg.losses.items():
            if loss_name in ['adv', 'feat']:
                for adv_name, _ in self.adv_losses.items():
                    loss_weights[f'{loss_name}_{adv_name}'] = weight
            elif weight > 0:
                self.aux_losses[loss_name] = builders.get_loss(loss_name, self.cfg)
                loss_weights[loss_name] = weight
            else:
                self.info_losses[loss_name] = builders.get_loss(loss_name, self.cfg)
        self.balancer = builders.get_balancer(loss_weights, self.cfg.balancer)
        self.register_stateful('adv_losses')

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        # best model is the last for the compression model
        return None

    def build_model(self):
        """Instantiate model and optimizer."""
        # Model and optimizer
        self.model = models.builders.get_compression_model(self.cfg).to(self.device)
        self.optimizer = builders.get_optimizer(self.model.parameters(), self.cfg.optim)
        self.register_stateful('model', 'optimizer')
        self.register_best_state('model')
        self.register_ema('model')

    def build_dataloaders(self):
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg)

    def show(self):
        """Show the compression model and employed adversarial loss."""
        self.logger.info(f"Compression model with {self.model.quantizer.total_codebooks} codebooks:")
        self.log_model_summary(self.model)
        self.logger.info("Adversarial loss:")
        self.log_model_summary(self.adv_losses)
        self.logger.info("Auxiliary losses:")
        self.logger.info(self.aux_losses)
        self.logger.info("Info losses:")
        self.logger.info(self.info_losses)

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        x = batch.to(self.device)
        y = x.clone()

        qres = self.model(x)
        assert isinstance(qres, quantization.QuantizedResult)
        y_pred = qres.x
        # Log bandwidth in kb/s
        metrics['bandwidth'] = qres.bandwidth.mean()

        if self.is_training:
            d_losses: dict = {}
            if len(self.adv_losses) > 0 and torch.rand(1, generator=self.rng).item() <= 1 / self.cfg.adversarial.every:
                for adv_name, adversary in self.adv_losses.items():
                    disc_loss = adversary.train_adv(y_pred, y)
                    d_losses[f'd_{adv_name}'] = disc_loss
                metrics['d_loss'] = torch.sum(torch.stack(list(d_losses.values())))
            metrics.update(d_losses)

        balanced_losses: dict = {}
        other_losses: dict = {}

        # penalty from quantization
        if qres.penalty is not None and qres.penalty.requires_grad:
            other_losses['penalty'] = qres.penalty  # penalty term from the quantizer

        # adversarial losses
        for adv_name, adversary in self.adv_losses.items():
            adv_loss, feat_loss = adversary(y_pred, y)
            balanced_losses[f'adv_{adv_name}'] = adv_loss
            balanced_losses[f'feat_{adv_name}'] = feat_loss

        # auxiliary losses
        for loss_name, criterion in self.aux_losses.items():
            loss = criterion(y_pred, y)
            balanced_losses[loss_name] = loss

        # weighted losses
        metrics.update(balanced_losses)
        metrics.update(other_losses)
        metrics.update(qres.metrics)

        if self.is_training:
            # backprop losses that are not handled by balancer
            other_loss = torch.tensor(0., device=self.device)
            if 'penalty' in other_losses:
                other_loss += other_losses['penalty']
            if other_loss.requires_grad:
                other_loss.backward(retain_graph=True)
                ratio1 = sum(p.grad.data.norm(p=2).pow(2)
                             for p in self.model.parameters() if p.grad is not None)
                assert isinstance(ratio1, torch.Tensor)
                metrics['ratio1'] = ratio1.sqrt()

            # balancer losses backward, returns effective training loss
            # with effective weights at the current batch.
            metrics['g_loss'] = self.balancer.backward(balanced_losses, y_pred)
            # add metrics corresponding to weight ratios
            metrics.update(self.balancer.metrics)
            ratio2 = sum(p.grad.data.norm(p=2).pow(2)
                         for p in self.model.parameters() if p.grad is not None)
            assert isinstance(ratio2, torch.Tensor)
            metrics['ratio2'] = ratio2.sqrt()

            # optim
            flashy.distrib.sync_model(self.model)
            if self.cfg.optim.max_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.max_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        # informative losses only
        info_losses: dict = {}
        with torch.no_grad():
            for loss_name, criterion in self.info_losses.items():
                loss = criterion(y_pred, y)
                info_losses[loss_name] = loss

        metrics.update(info_losses)

        # aggregated GAN losses: this is useful to report adv and feat across different adversarial loss setups
        adv_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('adv')]
        if len(adv_losses) > 0:
            metrics['adv'] = torch.sum(torch.stack(adv_losses))
        feat_losses = [loss for loss_name, loss in metrics.items() if loss_name.startswith('feat')]
        if len(feat_losses) > 0:
            metrics['feat'] = torch.sum(torch.stack(feat_losses))

        return metrics

    def run_epoch(self):
        # reset random seed at the beginning of the epoch
        self.rng = torch.Generator()
        self.rng.manual_seed(1234 + self.epoch)
        # run epoch
        super().run_epoch()

    def evaluate(self):
        """Evaluate stage. Runs audio reconstruction evaluation."""
        self.model.eval()
        evaluate_stage_name = str(self.current_stage)

        loader = self.dataloaders['evaluate']
        updates = len(loader)
        lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
        average = flashy.averager()

        pendings = []
        ctx = multiprocessing.get_context('spawn')
        with get_pool_executor(self.cfg.evaluate.num_workers, mp_context=ctx) as pool:
            for idx, batch in enumerate(lp):
                x = batch.to(self.device)
                with torch.no_grad():
                    qres = self.model(x)

                y_pred = qres.x.cpu()
                y = batch.cpu()  # should already be on CPU but just in case
                pendings.append(pool.submit(evaluate_audio_reconstruction, y_pred, y, self.cfg))

            metrics_lp = self.log_progress(f'{evaluate_stage_name} metrics', pendings, updates=self.log_updates)
            for pending in metrics_lp:
                metrics = pending.result()
                metrics = average(metrics)

        metrics = flashy.distrib.average_metrics(metrics, len(loader))
        return metrics

    def generate(self):
        """Generate stage."""
        self.model.eval()
        sample_manager = SampleManager(self.xp, map_reference_to_sample_id=True)
        generate_stage_name = str(self.current_stage)

        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        for batch in lp:
            reference, _ = batch
            reference = reference.to(self.device)
            with torch.no_grad():
                qres = self.model(reference)
            assert isinstance(qres, quantization.QuantizedResult)

            reference = reference.cpu()
            estimate = qres.x.cpu()
            sample_manager.add_samples(estimate, self.epoch, ground_truth_wavs=reference)

        flashy.distrib.barrier()

    def load_from_pretrained(self, name: str) -> dict:
        model = models.CompressionModel.get_pretrained(name)
        if isinstance(model, models.DAC):
            raise RuntimeError("Cannot fine tune a DAC model.")
        elif isinstance(model, models.HFEncodecCompressionModel):
            self.logger.warning('Trying to automatically convert a HuggingFace model '
                                'to AudioCraft, this might fail!')
            state = model.model.state_dict()
            new_state = {}
            for k, v in state.items():
                if k.startswith('decoder.layers') and '.conv.' in k and '.block.' not in k:
                    # We need to determine if this a convtr or a regular conv.
                    layer = int(k.split('.')[2])
                    if isinstance(model.model.decoder.layers[layer].conv, torch.nn.ConvTranspose1d):

                        k = k.replace('.conv.', '.convtr.')
                k = k.replace('encoder.layers.', 'encoder.model.')
                k = k.replace('decoder.layers.', 'decoder.model.')
                k = k.replace('conv.', 'conv.conv.')
                k = k.replace('convtr.', 'convtr.convtr.')
                k = k.replace('quantizer.layers.', 'quantizer.vq.layers.')
                k = k.replace('.codebook.', '._codebook.')
                new_state[k] = v
            state = new_state
        elif isinstance(model, models.EncodecModel):
            state = model.state_dict()
        else:
            raise RuntimeError(f"Cannot fine tune model type {type(model)}.")
        return {
            'best_state': {'model': state}
        }

    @staticmethod
    def model_from_checkpoint(checkpoint_path: tp.Union[Path, str],
                              device: tp.Union[torch.device, str] = 'cpu') -> models.CompressionModel:
        """Instantiate a CompressionModel from a given checkpoint path or dora sig.
        This method is a convenient endpoint to load a CompressionModel to use in other solvers.

        Args:
            checkpoint_path (Path or str): Path to checkpoint or dora sig from where the checkpoint is resolved.
                This also supports pre-trained models by using a path of the form //pretrained/NAME.
                See `model_from_pretrained` for a list of supported pretrained models.
            use_ema (bool): Use EMA variant of the model instead of the actual model.
            device (torch.device or str): Device on which the model is loaded.
        """
        checkpoint_path = str(checkpoint_path)
        if checkpoint_path.startswith('//pretrained/'):
            name = checkpoint_path.split('/', 3)[-1]
            return models.CompressionModel.get_pretrained(name, device)
        logger = logging.getLogger(__name__)
        logger.info(f"Loading compression model from checkpoint: {checkpoint_path}")
        _checkpoint_path = checkpoint.resolve_checkpoint_path(checkpoint_path, use_fsdp=False)
        assert _checkpoint_path is not None, f"Could not resolve compression model checkpoint path: {checkpoint_path}"
        state = checkpoint.load_checkpoint(_checkpoint_path)
        assert state is not None and 'xp.cfg' in state, f"Could not load compression model from ckpt: {checkpoint_path}"
        cfg = state['xp.cfg']
        cfg.device = device
        compression_model = models.builders.get_compression_model(cfg).to(device)
        assert compression_model.sample_rate == cfg.sample_rate, "Compression model sample rate should match"

        assert 'best_state' in state and state['best_state'] != {}
        assert 'exported' not in state, "When loading an exported checkpoint, use the //pretrained/ prefix."
        compression_model.load_state_dict(state['best_state']['model'])
        compression_model.eval()
        logger.info("Compression model loaded!")
        return compression_model

    @staticmethod
    def wrapped_model_from_checkpoint(cfg: omegaconf.DictConfig,
                                      checkpoint_path: tp.Union[Path, str],
                                      device: tp.Union[torch.device, str] = 'cpu') -> models.CompressionModel:
        """Instantiate a wrapped CompressionModel from a given checkpoint path or dora sig.

        Args:
            cfg (omegaconf.DictConfig): Configuration to read from for wrapped mode.
            checkpoint_path (Path or str): Path to checkpoint or dora sig from where the checkpoint is resolved.
            use_ema (bool): Use EMA variant of the model instead of the actual model.
            device (torch.device or str): Device on which the model is loaded.
        """
        compression_model = CompressionSolver.model_from_checkpoint(checkpoint_path, device)
        compression_model = models.builders.get_wrapped_compression_model(compression_model, cfg)
        return compression_model


def evaluate_audio_reconstruction(y_pred: torch.Tensor, y: torch.Tensor, cfg: omegaconf.DictConfig) -> dict:
    """Audio reconstruction evaluation method that can be conveniently pickled."""
    metrics = {}
    if cfg.evaluate.metrics.visqol:
        visqol = builders.get_visqol(cfg.metrics.visqol)
        metrics['visqol'] = visqol(y_pred, y, cfg.sample_rate)
    sisnr = builders.get_loss('sisnr', cfg)
    metrics['sisnr'] = sisnr(y_pred, y)
    return metrics
