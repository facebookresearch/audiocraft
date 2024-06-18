# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from functools import partial
import os
from pathlib import Path

import flashy
from omegaconf import DictConfig
import multiprocessing
import numpy as np
import torch
import torch.nn as nn

from . import base, builders
from ..models.builders import get_watermark_model
from ..modules.watermark import pad, mix

from ..metrics.miou import calculate_miou
from ..metrics.pesq import PesqMetric

from ..utils import checkpoint
from ..utils.audio_effects import (
    compress_with_encodec,
    get_audio_effects,
    select_audio_effects,
)
from ..utils.samples.manager import SampleManager
from ..data.audio import save_spectrograms
from ..utils.utils import get_pool_executor

from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


if tp.TYPE_CHECKING:
    from ..models.watermark import WMModel


def get_encodec_audio_effect(encodec_cfg: DictConfig, sr: int) -> tp.Dict:
    """
    Construct encodec-based compression data agumentation. This method is
    is put here instead of in `audiocraft.utils.audio_effects` because
    it depends on the package `audiocraft.solvers`, which is one layer
    higher than `audiocraft.utils`, so we avoid the circle dependency
    from any solvers using `audiocraft.utils.audio_effects` to do the
    augmentation
    """
    from ..solvers.compression import CompressionSolver

    codec_model = CompressionSolver.model_from_checkpoint(encodec_cfg.ckpt)
    codec_model.train()
    return {
        f"encodec_nq={n_q}": partial(
            compress_with_encodec,
            model=codec_model,
            n_q=n_q,
            sample_rate=sr,
        )
        for n_q in encodec_cfg.n_qs
    }


def random_message(nbits: int, batch_size: int) -> torch.Tensor:
    """Return random message as 0/1 tensor."""
    if nbits == 0:
        return torch.tensor([])
    return torch.randint(0, 2, (batch_size, nbits))


class WatermarkSolver(base.StandardSolver):
    """Solver for different watermarking models"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.rng: torch.Generator  # set at each epoch
        self.model: WMModel
        if hasattr(cfg, "fsdp"):
            assert not getattr(
                cfg.fsdp, "use", False
            ), "FSDP not supported by WatermarkSolver."
        self._init_losses()
        self._init_augmentations()
        self.balancer = builders.get_balancer(self.loss_weights, self.cfg.balancer)
        self.path_specs = os.path.join(self.folder, "spectrograms")
        os.makedirs(self.path_specs, exist_ok=True)

    def _init_losses(self):
        assert hasattr(self.cfg, "losses") and isinstance(
            self.cfg.losses, (DictConfig, tp.Mapping)
        ), "WatermarkSolver must declare training losses in the config"

        self.adv_losses = builders.get_adversarial_losses(self.cfg)  # noqa
        self.register_stateful("adv_losses")

        self.aux_losses = nn.ModuleDict()  # noqa
        self.info_losses = nn.ModuleDict()  # noqa
        self.wm_losses = nn.ModuleDict()  # noqa
        loss_weights = {}
        for loss_name, weight in self.cfg.losses.items():

            # explicitly skip this loss calculation by setting a -1 as weight
            # if weight == 0 it will be calculated but kept as info
            if weight == -1:
                continue

            if loss_name in ["adv", "feat"]:
                for adv_name, _ in self.adv_losses.items():
                    loss_weights[f"{loss_name}_{adv_name}"] = weight
            elif weight > 0:
                if loss_name[:3] == "wm_":
                    self.wm_losses[loss_name] = builders.get_loss(
                        loss_name, self.cfg
                    ).to(self.device)
                    loss_weights[loss_name] = weight
                else:
                    self.aux_losses[loss_name] = builders.get_loss(
                        loss_name, self.cfg
                    ).to(self.device)
                    loss_weights[loss_name] = weight
            else:
                self.info_losses[loss_name] = builders.get_loss(loss_name, self.cfg).to(
                    self.device
                )

        self.loss_weights = loss_weights  # noqa

    def _init_augmentations(self):
        if not hasattr(self.cfg, "aug_weights") or not hasattr(
            self.cfg, "audio_effects"
        ):
            return

        aug_weights = {}
        cfg_audio_effects = dict(self.cfg.audio_effects)

        # Handle `encodec` augmentation separately as this requires loading a
        # CompressionSolver checkpoint
        encodec_cfg = cfg_audio_effects.pop("encodec", None)
        if encodec_cfg:
            encodec_effects = get_encodec_audio_effect(
                encodec_cfg, self.cfg.sample_rate
            )
            for aug_name in encodec_effects.keys():
                aug_weights[aug_name] = getattr(self.cfg.aug_weights, "encodec", -1)
        else:
            encodec_effects = {}

        other_effects = get_audio_effects(self.cfg)  # noqa
        for name in other_effects.keys():
            aug_weights[name] = self.cfg.aug_weights.get(name, -1)

        self.aug_weights = aug_weights  # noqa
        self.augmentations = {**encodec_effects, **other_effects}  # noqa

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        # best model is the last for the watermark model for now
        return None

    def build_model(self):
        """Instantiate model and optimizer."""
        # Model and optimizer
        self.model = get_watermark_model(self.cfg)
        # Need two optimizers ?
        self.optimizer = builders.get_optimizer(self.model.parameters(), self.cfg.optim)
        self.register_stateful("model", "optimizer")
        self.register_best_state("model")
        self.register_ema("model")

    def build_dataloaders(self):
        """Instantiate audio dataloaders for each stage."""
        self.dataloaders = builders.get_audio_datasets(self.cfg)

    def show(self):
        """Show the Watermark model and employed adversarial loss."""
        self.log_model_summary(self.model)
        self.logger.info("Sould print losses here:")

    def crop(
        self, signal: torch.Tensor, watermark: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies a transformation to modify the watermarked signal to train localization.
        It can be one of the following:
            - zero padding: add zeros at the begining and the end of the signal
            - crop: crop the watermark apply a watermark only on some parts of the signal
            - shuffle: replace some part of the audio with other non watermarked parts
                from the batch
        In every cases the function returns a mask that contains indicates the parts that are or
        not watermarked

        Args:
            watermark (torch.Tensor): The watermark to apply on the signal.
            signal (torch.Tensor): clean signal
        Returns:
            watermark (torch.Tensor): modified watermark
            signal (torch.Tensor): modified signal
            mask (torch.Tensor): mask indicating which portion is still watermarked
        """
        assert (
            self.cfg.crop.prob + self.cfg.crop.shuffle_prob + self.cfg.crop.pad_prob
            <= 1
        ), f"The sum of the probabilities {self.cfg.crop.prob=} {self.cfg.crop.shuffle_prob=} \
                {self.cfg.crop.pad_prob=} should be less than 1"
        mask = torch.ones_like(watermark)
        p = torch.rand(1)
        if p < self.cfg.crop.pad_prob:  # Pad with some probability
            start = int(torch.rand(1) * 0.33 * watermark.size(-1))
            finish = int((0.66 + torch.rand(1) * 0.33) * watermark.size(-1))
            mask[:, :, :start] = 0
            mask[:, :, finish:] = 0
            if torch.rand(1) > 0.5:
                mask = 1 - mask
            signal *= mask  # pad signal

        elif (
            p < self.cfg.crop.prob + self.cfg.crop.pad_prob + self.cfg.crop.shuffle_prob
        ):
            # Define a mask, then crop or shuffle
            mask_size = round(watermark.shape[-1] * self.cfg.crop.size)
            n_windows = int(
                torch.randint(1, self.cfg.crop.max_n_windows + 1, (1,)).item()
            )
            window_size = int(mask_size / n_windows)
            for _ in range(n_windows):  # Create multiple windows in the mask
                mask_start = torch.randint(0, watermark.shape[-1] - window_size, (1,))
                mask[:, :, mask_start: mask_start + window_size] = (
                    0  # Apply window to mask
                )
            # inverse the mask half the time
            if torch.rand(1) > 0.5:
                mask = 1 - mask

            if p < self.cfg.crop.pad_prob + self.cfg.crop.shuffle_prob:  # shuffle
                # shuffle
                signal_cloned = signal.clone().detach()  # detach to be sure
                shuffle_idx = torch.randint(0, signal.size(0), (signal.size(0),))
                signal = signal * mask + signal_cloned[shuffle_idx] * (
                    1 - mask
                )  # shuffle signal where not wm

        watermark *= mask  # Apply mask to the watermark
        return signal, watermark, mask

    def run_step(self, idx: int, batch: torch.Tensor, metrics: dict):
        """Perform one training or valid step on a given batch."""
        x = batch.to(self.device)
        y = x.clone()
        nbits = getattr(self.model, "nbits")
        message = random_message(nbits, y.shape[0]).to(self.device)
        watermark = self.model.get_watermark(x, message=message)
        y, watermark, mask = self.crop(y, watermark)

        y_wm = y + watermark

        if (
            self.cfg.losses.adv != 0 or self.cfg.losses.feat != 0
        ) and self.is_training:  # train quality adv
            d_losses: dict = {}
            if (
                len(self.adv_losses) > 0
                and torch.rand(1, generator=self.rng).item()
                <= 1 / self.cfg.adversarial.every
            ):
                for adv_name, adversary in self.adv_losses.items():
                    disc_loss = adversary.train_adv(y_wm, y)
                    d_losses[f"d_{adv_name}"] = disc_loss
                metrics["d_loss"] = torch.sum(torch.stack(list(d_losses.values())))
            metrics.update(d_losses)

        balanced_losses: dict = {}
        other_losses: dict = {}

        # adversarial losses
        if self.cfg.losses.adv != 0 or self.cfg.losses.feat != 0:
            for adv_name, adversary in self.adv_losses.items():
                adv_loss, feat_loss = adversary(y_wm, y)
                balanced_losses[f"adv_{adv_name}"] = adv_loss
                balanced_losses[f"feat_{adv_name}"] = feat_loss

        # auxiliary losses on quality/similarity
        for loss_name, criterion in self.aux_losses.items():
            loss = criterion(y_wm, y)
            balanced_losses[loss_name] = loss

        # apply augmentations
        mode = "all" if self.cfg.select_aug_mode == "all" else "weighted"
        selected_augs = select_audio_effects(
            self.augmentations,
            self.aug_weights,
            mode=mode,
            max_length=self.cfg.n_max_aug,
        )
        N_augs = len(selected_augs)
        for (
            augmentation_name,
            augmentation_method,
        ) in selected_augs.items():
            # concatenate to use the augmentation function only once
            y_y_wm = torch.cat([y, y_wm], dim=0)
            aug_cat, mask_aug = augmentation_method(y_y_wm, mask=mask)
            aug_y = aug_cat[: y.size(0)]
            aug_y_wm = aug_cat[y.size(0):]
            positive = self.model.detect_watermark(aug_y_wm)
            negative = self.model.detect_watermark(aug_y)
            for loss_name, criterion in self.wm_losses.items():
                loss = criterion(positive, negative, mask_aug, message)
                other_losses[f"{loss_name}_{augmentation_name}"] = loss

        # weighted losses
        metrics.update(balanced_losses)
        metrics.update(other_losses)
        if self.is_training:  # something is weird about the loss balancer not
            other_loss = torch.tensor(0.0, device=self.device)
            for name, o_loss in other_losses.items():
                if "wm_detection" in name:
                    # here we include the detection losses for augmentation
                    other_loss += (self.loss_weights["wm_detection"] / N_augs) * o_loss
                elif "wm_mb" in name:
                    other_loss += (self.loss_weights["wm_mb"] / N_augs) * o_loss
                else:
                    other_loss += self.loss_weights[name] * o_loss
            if other_loss.requires_grad:
                other_loss.backward(retain_graph=True)
                ratio1 = sum(
                    p.grad.data.norm(p=2).pow(2)
                    for p in self.model.parameters()
                    if p.grad is not None
                )
                assert isinstance(ratio1, torch.Tensor)
                metrics["ratio1"] = ratio1.sqrt()

            # balancer losses backward, returns effective training loss
            # with effective weights at the current batch.
            metrics["g_loss"] = self.balancer.backward(balanced_losses, y_wm)
            # add metrics corresponding to weight ratios
            metrics.update(self.balancer.metrics)
            ratio2 = sum(
                p.grad.data.norm(p=2).pow(2)
                for p in self.model.parameters()
                if p.grad is not None
            )
            assert isinstance(ratio2, torch.Tensor)
            metrics["ratio2"] = ratio2.sqrt()

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
                loss = criterion(y_wm, y)
                info_losses[loss_name] = loss
            # pesq
            metrics["pesq"] = tensor_pesq(y_wm, y, sr=self.cfg.sample_rate)
            # max allocated memory
            metrics["max_mem"] = torch.cuda.max_memory_allocated() / 1e9

        metrics.update(info_losses)
        if self.cfg.losses.adv != 0 or self.cfg.losses.feat != 0:
            # aggregated GAN losses: this is useful to report adv and feat across different adversarial loss setups
            adv_losses = [
                loss
                for loss_name, loss in metrics.items()
                if loss_name.startswith("adv")
            ]
            if len(adv_losses) > 0:
                metrics["adv"] = torch.sum(torch.stack(adv_losses))
            feat_losses = [
                loss
                for loss_name, loss in metrics.items()
                if loss_name.startswith("feat")
            ]
            if len(feat_losses) > 0:
                metrics["feat"] = torch.sum(torch.stack(feat_losses))

        return metrics

    def run_epoch(self):
        # reset random seed at the beginning of the epoch
        self.rng = torch.Generator()
        self.rng.manual_seed(1234 + self.epoch)
        # run epoch
        super().run_epoch()

    def evaluate(self) -> dict:
        """Evaluate stage. Runs audio reconstruction evaluation."""
        self.model.eval()
        evaluate_stage_name = str(self.current_stage)

        loader = self.dataloaders["evaluate"]
        updates = len(loader)
        lp = self.log_progress(
            f"{evaluate_stage_name} inference",
            loader,
            total=updates,
            updates=self.log_updates,
        )
        average = flashy.averager()

        pendings = []
        ctx = multiprocessing.get_context("spawn")
        with get_pool_executor(self.cfg.evaluate.num_workers, mp_context=ctx) as pool:
            for batch in lp:
                x = batch.to(self.device)
                with torch.no_grad():
                    message = random_message(self.model.nbits, x.shape[0])
                    watermark = self.model.get_watermark(x, message)
                    x_wm = x + watermark
                y_pred = x_wm.cpu()
                y = batch.cpu()  # should already be on CPU but just in case
                pendings.append(
                    pool.submit(
                        evaluate_audio_watermark,
                        y_pred,
                        y,
                        self.cfg,
                    )
                )
                # evaluate augmentations
                # evaluation is run on all the augmentations
                for (
                    augmentation_name,
                    augmentation_method,
                ) in self.augmentations.items():
                    # if (
                    #     "mp3" in augmentation_name
                    #     and idx >= 8
                    #     and self.cfg.evaluate.every <= 2
                    # ):
                    #     # When evaluating often do not compute mp3 on the full eval dset to make things faster
                    #     continue
                    with torch.no_grad():
                        aug_positive = self.model.detect_watermark(
                            augmentation_method(x_wm)
                        )
                        aug_negative = self.model.detect_watermark(
                            augmentation_method(x)
                        )

                    pendings.append(
                        pool.submit(
                            evaluate_augmentations,
                            aug_positive.cpu(),
                            aug_negative.cpu(),
                            augmentation_name,
                            message.cpu(),
                        )
                    )
                # end eval of augmentations

                # evaluate localization cropping
                for window_size in np.linspace(0.1, 0.9, 9):

                    mixed, true_predictions = mix(x, x_wm, window_size=window_size)
                    model_predictions = self.model.detect_watermark(mixed)
                    pendings.append(
                        pool.submit(
                            evaluate_localizations,
                            model_predictions.cpu(),
                            true_predictions.cpu(),
                            f"crop_{window_size:0.1f}",
                        )
                    )
                    mixed, true_predictions = mix(
                        x, x_wm, window_size=window_size, shuffle=True
                    )
                    model_predictions = self.model.detect_watermark(mixed)
                    pendings.append(
                        pool.submit(
                            evaluate_localizations,
                            model_predictions.cpu(),
                            true_predictions.cpu(),
                            f"shuffle_{window_size:0.1f}",
                        )
                    )
                # evaluate localization padding
                mixed, true_predictions = pad(x_wm)
                model_predictions = self.model.detect_watermark(mixed)
                pendings.append(
                    pool.submit(
                        evaluate_localizations,
                        model_predictions.cpu(),
                        true_predictions.cpu(),
                        "padding",
                    )
                )
                mixed, true_predictions = pad(x_wm, central=True)
                model_predictions = self.model.detect_watermark(mixed)
                pendings.append(
                    pool.submit(
                        evaluate_localizations,
                        model_predictions.cpu(),
                        true_predictions.cpu(),
                        "central_padding",
                    )
                )
                # end of evaluate localization

            metrics_lp = self.log_progress(
                f"{evaluate_stage_name} metrics", pendings, updates=self.log_updates
            )
            for pending in metrics_lp:
                metrics = pending.result()
                metrics = average(metrics)

        metrics = flashy.distrib.average_metrics(metrics, len(loader))
        if self.cfg.select_aug_mode == "use_eval_acc":
            # Adjust augmentation weights based on evaluation loss.
            # Higher accuracy results in lower probability of selecting this augmentation.
            for name in self.augmentations.keys():
                if (
                    self.aug_weights[name] != -1
                ):  # keep weight to -1 for unwanted augmentations
                    # set to 0.05 to ensure that an augmentation is never completely removed during a full epoch.
                    self.aug_weights[name] = max(1 - metrics[f"aug_{name}_acc"], 0.05)
        return metrics

    def generate(self):
        """Generate stage."""
        self.model.eval()
        sample_manager = SampleManager(self.xp, map_reference_to_sample_id=True)
        generate_stage_name = str(self.current_stage)

        loader = self.dataloaders["generate"]
        updates = len(loader)
        lp = self.log_progress(
            generate_stage_name, loader, total=updates, updates=self.log_updates
        )
        path_dir = os.path.join(self.path_specs, f"epoch={self.epoch}")
        os.makedirs(path_dir, exist_ok=True)
        first_batch = True
        for batch in lp:
            reference, _ = batch
            reference = reference.to(self.device)
            with torch.no_grad():
                message = random_message(self.model.nbits, reference.shape[0])
                watermark = self.model.get_watermark(reference, message)
                x_wm = reference + watermark

            reference = reference.cpu()
            sample_manager.add_samples(
                x_wm.cpu(), self.epoch, ground_truth_wavs=reference
            )
            if first_batch and flashy.distrib.is_rank_zero():
                for i in range(reference.size(0)):
                    ys = [
                        reference.cpu()[i].squeeze(0).numpy(),
                        x_wm.cpu()[i].squeeze(0).numpy(),
                        watermark.cpu()[i].squeeze(0).numpy(),
                    ]
                    path = os.path.join(path_dir, f"spec_{i}.pdf")
                    save_spectrograms(
                        ys,
                        names=["Ground Truth", "Audio Watermarked", "Watermark"],
                        sr=self.cfg.sample_rate,
                        path=path,
                    )
                first_batch = False
        flashy.distrib.barrier()

    def load_from_pretrained(self, name: str) -> dict:
        raise ValueError("No pretrained model")

    @staticmethod
    def model_from_checkpoint(
        checkpoint_path: tp.Union[Path, str],
        device: tp.Union[torch.device, str] = "cpu",
    ) -> "WMModel":
        """Instantiate a WatermarkModel from a given checkpoint path or dora sig.

        Args:
            checkpoint_path (Path or str): Path to checkpoint or dora sig from where the checkpoint is resolved.
            device (torch.device or str): Device on which the model is loaded.
        """
        checkpoint_path = str(checkpoint_path)
        logger = logging.getLogger(__name__)
        logger.info(f"Loading WatermarkModel from checkpoint: {checkpoint_path}")
        _checkpoint_path = checkpoint.resolve_checkpoint_path(
            checkpoint_path, use_fsdp=False
        )
        assert (
            _checkpoint_path is not None
        ), f"Could not resolve WatermarkModel checkpoint path: {checkpoint_path}"
        state = checkpoint.load_checkpoint(_checkpoint_path)
        assert (
            state is not None and "xp.cfg" in state
        ), f"Could not load WatermarkModel from ckpt: {checkpoint_path}"
        cfg = state["xp.cfg"]
        cfg.device = device
        watermarking_model = get_watermark_model(cfg).to(device)

        assert "best_state" in state and state["best_state"] != {}
        assert (
            "exported" not in state
        ), "When loading an exported checkpoint, use the //pretrained/ prefix."
        watermarking_model.load_state_dict(state["best_state"]["model"])
        watermarking_model.eval()
        logger.info("Watermarking model loaded!")
        return watermarking_model


def evaluate_localizations(predictions, true_predictions, name):
    metrics = {}
    # predictions are output of the detector shape [bsz, 2, frames]
    # true_predictions is output of the mix method shape [bsz, 2, frames]
    metrics[f"localization_acc_{name}"] = (
        ((predictions[:, 1, :] > 0.5) == true_predictions[:, 1, :])
        .float()
        .mean()
        .item()
    )
    metrics[f"localization_miou_{name}"] = calculate_miou(
        predictions[:, 1, :], true_predictions[:, 1, :]
    )
    return metrics


def evaluate_augmentations(
    positive: torch.Tensor,
    negative: torch.Tensor,
    augmentation_name: str,
    message: torch.Tensor,
) -> dict:
    """calculating evaluation metrics but take name of the augmentation
    method that has been done before getting positive and negative results"""
    metrics = {}
    metrics[f"aug_{augmentation_name}_acc"] = compute_accuracy(positive, negative)
    metrics[f"aug_{augmentation_name}_fpr"] = compute_FPR(negative)
    metrics[f"aug_{augmentation_name}_fnr"] = compute_FNR(positive)
    if message.shape[0] != 0:
        metrics[f"aug_{augmentation_name}_bit_acc"] = compute_bit_acc(positive, message)

    # add one metric which is average overall score of all augmentations
    metrics["all_aug_acc"] = compute_accuracy(positive, negative)

    return metrics


def evaluate_audio_watermark(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    cfg: DictConfig,
) -> dict:
    """Audio reconstruction evaluation method that can be conveniently pickled."""
    metrics = {}
    if cfg.evaluate.metrics.visqol:
        visqol = builders.get_visqol(cfg.metrics.visqol)
        metrics["visqol"] = visqol(y_pred, y, cfg.sample_rate)
    sisnr = ScaleInvariantSignalNoiseRatio().to(y.device)
    stoi = ShortTimeObjectiveIntelligibility(fs=cfg.sample_rate)
    metrics["sisnr"] = sisnr(y_pred, y)
    metrics["stoi"] = stoi(y_pred, y)
    metrics["pesq"] = tensor_pesq(y_pred, y, sr=cfg.sample_rate)
    return metrics


def tensor_pesq(y_pred: torch.Tensor, y: torch.Tensor, sr: int):
    # pesq returns error if no speech is detected, so we catch it
    return PesqMetric(sr)(y_pred, y).item()


def compute_accuracy(positive, negative):
    N = (positive[:, 1, :].mean(dim=1) > 0.5).sum() + (
        negative[:, 0, :].mean(dim=1) > 0.5
    ).sum()
    acc = N / (2 * positive.size(0))
    return acc


def compute_FPR(negative):
    N = (negative[:, 1, :].mean(dim=1) > 0.5).sum()
    fpr = N / (negative.size(0))
    return fpr


def compute_FNR(positive):
    N = (positive[:, 0, :].mean(dim=1) > 0.5).sum()
    fpr = N / (positive.size(0))
    return fpr


def _bit_acc(decoded, original):
    bit_acc = (decoded == original).float().mean()
    return bit_acc


def compute_bit_acc(positive, original, mask=None):
    """Compute bit accuracy.
    Args:
        positive: detector outputs [bsz, 2+nbits, time_steps]
        original: original message (0 or 1) [bsz, nbits]
        mask: mask of the watermark [bsz, 1, time_steps]
    """
    decoded = positive[:, 2:, :]  # b 2+nbits t -> b nbits t
    if mask is not None:
        # cut last dim of positive to keep only where mask is 1
        new_shape = [*decoded.shape[:-1], -1]  # b nbits t -> b nbits -1
        decoded = torch.masked_select(decoded, mask == 1).reshape(new_shape)
    # average decision over time, then threshold
    decoded = decoded.mean(dim=-1) > 0  # b nbits
    return _bit_acc(decoded, original)
