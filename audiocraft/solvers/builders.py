# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
All the functions to build the relevant solvers and used objects
from the Hydra config.
"""

from enum import Enum
import logging
import typing as tp

import dora
import flashy
import omegaconf
import torch
from torch import nn
from torch.optim import Optimizer
# LRScheduler was renamed in some torch versions
try:
    from torch.optim.lr_scheduler import LRScheduler  # type: ignore
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .base import StandardSolver
from .. import adversarial, data, losses, metrics, optim
from ..utils.utils import dict_from_config, get_loader


logger = logging.getLogger(__name__)


class DatasetType(Enum):
    AUDIO = "audio"
    MUSIC = "music"
    SOUND = "sound"


def get_solver(cfg: omegaconf.DictConfig) -> StandardSolver:
    """Instantiate solver from config."""
    from .audiogen import AudioGenSolver
    from .compression import CompressionSolver
    from .musicgen import MusicGenSolver
    from .diffusion import DiffusionSolver
    from .magnet import MagnetSolver, AudioMagnetSolver
    klass = {
        'compression': CompressionSolver,
        'musicgen': MusicGenSolver,
        'audiogen': AudioGenSolver,
        'magnet': MagnetSolver,
        'audio_magnet': AudioMagnetSolver,
        'lm': MusicGenSolver,  # backward compatibility
        'diffusion': DiffusionSolver,
        'sound_lm': AudioGenSolver,  # backward compatibility
    }[cfg.solver]
    return klass(cfg)  # type: ignore


def get_optim_parameter_groups(model: nn.Module):
    """Create parameter groups for the model using the appropriate method
    if defined for each modules, to create the different groups.

    Args:
        model (nn.Module): torch model
    Returns:
        List of parameter groups
    """
    seen_params: tp.Set[nn.parameter.Parameter] = set()
    other_params = []
    groups = []
    for name, module in model.named_modules():
        if hasattr(module, 'make_optim_group'):
            group = module.make_optim_group()
            params = set(group['params'])
            assert params.isdisjoint(seen_params)
            seen_params |= set(params)
            groups.append(group)
    for param in model.parameters():
        if param not in seen_params:
            other_params.append(param)
    groups.insert(0, {'params': other_params})
    parameters = groups
    return parameters


def get_optimizer(params: tp.Union[nn.Module, tp.Iterable[torch.Tensor]], cfg: omegaconf.DictConfig) -> Optimizer:
    """Build torch optimizer from config and set of parameters.
    Supported optimizers: Adam, AdamW

    Args:
        params (nn.Module or iterable of torch.Tensor): Parameters to optimize.
        cfg (DictConfig): Optimization-related configuration.
    Returns:
        torch.optim.Optimizer.
    """
    if 'optimizer' not in cfg:
        if getattr(cfg, 'optim', None) is not None:
            raise KeyError("Optimizer not found in config. Try instantiating optimizer from cfg.optim?")
        else:
            raise KeyError("Optimizer not found in config.")

    parameters = get_optim_parameter_groups(params) if isinstance(params, nn.Module) else params
    optimizer: torch.optim.Optimizer
    if cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr, **cfg.adam)
    elif cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, **cfg.adam)
    elif cfg.optimizer == 'dadam':
        optimizer = optim.DAdaptAdam(parameters, lr=cfg.lr, **cfg.adam)
    else:
        raise ValueError(f"Unsupported Optimizer: {cfg.optimizer}")
    return optimizer


def get_lr_scheduler(optimizer: torch.optim.Optimizer,
                     cfg: omegaconf.DictConfig,
                     total_updates: int) -> tp.Optional[LRScheduler]:
    """Build torch learning rate scheduler from config and associated optimizer.
    Supported learning rate schedulers: ExponentialLRScheduler, PlateauLRScheduler

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        cfg (DictConfig): Schedule-related configuration.
        total_updates (int): Total number of updates.
    Returns:
        torch.optim.Optimizer.
    """
    if 'lr_scheduler' not in cfg:
        raise KeyError("LR Scheduler not found in config")

    lr_sched: tp.Optional[LRScheduler] = None
    if cfg.lr_scheduler == 'step':
        lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, **cfg.step)
    elif cfg.lr_scheduler == 'exponential':
        lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.exponential)
    elif cfg.lr_scheduler == 'cosine':
        kwargs = dict_from_config(cfg.cosine)
        warmup_steps = kwargs.pop('warmup')
        lr_sched = optim.CosineLRScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_updates, **kwargs)
    elif cfg.lr_scheduler == 'polynomial_decay':
        kwargs = dict_from_config(cfg.polynomial_decay)
        warmup_steps = kwargs.pop('warmup')
        lr_sched = optim.PolynomialDecayLRScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_updates, **kwargs)
    elif cfg.lr_scheduler == 'inverse_sqrt':
        kwargs = dict_from_config(cfg.inverse_sqrt)
        warmup_steps = kwargs.pop('warmup')
        lr_sched = optim.InverseSquareRootLRScheduler(optimizer, warmup_steps=warmup_steps, **kwargs)
    elif cfg.lr_scheduler == 'linear_warmup':
        kwargs = dict_from_config(cfg.linear_warmup)
        warmup_steps = kwargs.pop('warmup')
        lr_sched = optim.LinearWarmupLRScheduler(optimizer, warmup_steps=warmup_steps, **kwargs)
    elif cfg.lr_scheduler is not None:
        raise ValueError(f"Unsupported LR Scheduler: {cfg.lr_scheduler}")
    return lr_sched


def get_ema(module_dict: nn.ModuleDict, cfg: omegaconf.DictConfig) -> tp.Optional[optim.ModuleDictEMA]:
    """Initialize Exponential Moving Average.

    Args:
        module_dict (nn.ModuleDict): ModuleDict for which to compute the EMA.
        cfg (omegaconf.DictConfig): Optim EMA configuration.
    Returns:
        optim.ModuleDictEMA: EMA version of the ModuleDict.
    """
    kw: tp.Dict[str, tp.Any] = dict(cfg)
    use = kw.pop('use', False)
    decay = kw.pop('decay', None)
    device = kw.pop('device', None)
    if not use:
        return None
    if len(module_dict) == 0:
        raise ValueError("Trying to build EMA but an empty module_dict source is provided!")
    ema_module = optim.ModuleDictEMA(module_dict, decay=decay, device=device)
    return ema_module


def get_loss(loss_name: str, cfg: omegaconf.DictConfig):
    """Instantiate loss from configuration."""
    klass = {
        'l1': torch.nn.L1Loss,
        'l2': torch.nn.MSELoss,
        'mel': losses.MelSpectrogramL1Loss,
        'mrstft': losses.MRSTFTLoss,
        'msspec': losses.MultiScaleMelSpectrogramLoss,
        'sisnr': losses.SISNR,
    }[loss_name]
    kwargs = dict(getattr(cfg, loss_name))
    return klass(**kwargs)


def get_balancer(loss_weights: tp.Dict[str, float], cfg: omegaconf.DictConfig) -> losses.Balancer:
    """Instantiate loss balancer from configuration for the provided weights."""
    kwargs: tp.Dict[str, tp.Any] = dict_from_config(cfg)
    return losses.Balancer(loss_weights, **kwargs)


def get_adversary(name: str, cfg: omegaconf.DictConfig) -> nn.Module:
    """Initialize adversary from config."""
    klass = {
        'msd': adversarial.MultiScaleDiscriminator,
        'mpd': adversarial.MultiPeriodDiscriminator,
        'msstftd': adversarial.MultiScaleSTFTDiscriminator,
    }[name]
    adv_cfg: tp.Dict[str, tp.Any] = dict(getattr(cfg, name))
    return klass(**adv_cfg)


def get_adversarial_losses(cfg) -> nn.ModuleDict:
    """Initialize dict of adversarial losses from config."""
    device = cfg.device
    adv_cfg = getattr(cfg, 'adversarial')
    adversaries = adv_cfg.get('adversaries', [])
    adv_loss_name = adv_cfg['adv_loss']
    feat_loss_name = adv_cfg.get('feat_loss')
    normalize = adv_cfg.get('normalize', True)
    feat_loss: tp.Optional[adversarial.FeatureMatchingLoss] = None
    if feat_loss_name:
        assert feat_loss_name in ['l1', 'l2'], f"Feature loss only support L1 or L2 but {feat_loss_name} found."
        loss = get_loss(feat_loss_name, cfg)
        feat_loss = adversarial.FeatureMatchingLoss(loss, normalize)
    loss = adversarial.get_adv_criterion(adv_loss_name)
    loss_real = adversarial.get_real_criterion(adv_loss_name)
    loss_fake = adversarial.get_fake_criterion(adv_loss_name)
    adv_losses = nn.ModuleDict()
    for adv_name in adversaries:
        adversary = get_adversary(adv_name, cfg).to(device)
        optimizer = get_optimizer(adversary.parameters(), cfg.optim)
        adv_loss = adversarial.AdversarialLoss(
            adversary,
            optimizer,
            loss=loss,
            loss_real=loss_real,
            loss_fake=loss_fake,
            loss_feat=feat_loss,
            normalize=normalize
        )
        adv_losses[adv_name] = adv_loss
    return adv_losses


def get_visqol(cfg: omegaconf.DictConfig) -> metrics.ViSQOL:
    """Instantiate ViSQOL metric from config."""
    kwargs = dict_from_config(cfg)
    return metrics.ViSQOL(**kwargs)


def get_fad(cfg: omegaconf.DictConfig) -> metrics.FrechetAudioDistanceMetric:
    """Instantiate Frechet Audio Distance metric from config."""
    kwargs = dict_from_config(cfg.tf)
    xp = dora.get_xp()
    kwargs['log_folder'] = xp.folder
    return metrics.FrechetAudioDistanceMetric(**kwargs)


def get_kldiv(cfg: omegaconf.DictConfig) -> metrics.KLDivergenceMetric:
    """Instantiate KL-Divergence metric from config."""
    kld_metrics = {
        'passt': metrics.PasstKLDivergenceMetric,
    }
    klass = kld_metrics[cfg.model]
    kwargs = dict_from_config(cfg.get(cfg.model))
    return klass(**kwargs)


def get_text_consistency(cfg: omegaconf.DictConfig) -> metrics.TextConsistencyMetric:
    """Instantiate Text Consistency metric from config."""
    text_consistency_metrics = {
        'clap': metrics.CLAPTextConsistencyMetric
    }
    klass = text_consistency_metrics[cfg.model]
    kwargs = dict_from_config(cfg.get(cfg.model))
    return klass(**kwargs)


def get_chroma_cosine_similarity(cfg: omegaconf.DictConfig) -> metrics.ChromaCosineSimilarityMetric:
    """Instantiate Chroma Cosine Similarity metric from config."""
    assert cfg.model == 'chroma_base', "Only support 'chroma_base' method for chroma cosine similarity metric"
    kwargs = dict_from_config(cfg.get(cfg.model))
    return metrics.ChromaCosineSimilarityMetric(**kwargs)


def get_audio_datasets(cfg: omegaconf.DictConfig,
                       dataset_type: DatasetType = DatasetType.AUDIO) -> tp.Dict[str, torch.utils.data.DataLoader]:
    """Build AudioDataset from configuration.

    Args:
        cfg (omegaconf.DictConfig): Configuration.
        dataset_type: The type of dataset to create.
    Returns:
        dict[str, torch.utils.data.DataLoader]: Map of dataloader for each data split.
    """
    dataloaders: dict = {}

    sample_rate = cfg.sample_rate
    channels = cfg.channels
    seed = cfg.seed
    max_sample_rate = cfg.datasource.max_sample_rate
    max_channels = cfg.datasource.max_channels

    assert cfg.dataset is not None, "Could not find dataset definition in config"

    dataset_cfg = dict_from_config(cfg.dataset)
    splits_cfg: dict = {}
    splits_cfg['train'] = dataset_cfg.pop('train')
    splits_cfg['valid'] = dataset_cfg.pop('valid')
    splits_cfg['evaluate'] = dataset_cfg.pop('evaluate')
    splits_cfg['generate'] = dataset_cfg.pop('generate')
    execute_only_stage = cfg.get('execute_only', None)

    for split, path in cfg.datasource.items():
        if not isinstance(path, str):
            continue  # skipping this as not a path
        if execute_only_stage is not None and split != execute_only_stage:
            continue
        logger.info(f"Loading audio data split {split}: {str(path)}")
        assert (
            cfg.sample_rate <= max_sample_rate
        ), f"Expecting a max sample rate of {max_sample_rate} for datasource but {sample_rate} found."
        assert (
            cfg.channels <= max_channels
        ), f"Expecting a max number of channels of {max_channels} for datasource but {channels} found."

        split_cfg = splits_cfg[split]
        split_kwargs = {k: v for k, v in split_cfg.items()}
        kwargs = {**dataset_cfg, **split_kwargs}  # split kwargs overrides default dataset_cfg
        kwargs['sample_rate'] = sample_rate
        kwargs['channels'] = channels

        if kwargs.get('permutation_on_files') and cfg.optim.updates_per_epoch:
            kwargs['num_samples'] = (
                flashy.distrib.world_size() * cfg.dataset.batch_size * cfg.optim.updates_per_epoch)

        num_samples = kwargs['num_samples']
        shuffle = kwargs['shuffle']

        return_info = kwargs.pop('return_info')
        batch_size = kwargs.pop('batch_size', None)
        num_workers = kwargs.pop('num_workers')

        if dataset_type == DatasetType.MUSIC:
            dataset = data.music_dataset.MusicDataset.from_meta(path, **kwargs)
        elif dataset_type == DatasetType.SOUND:
            dataset = data.sound_dataset.SoundDataset.from_meta(path, **kwargs)
        elif dataset_type == DatasetType.AUDIO:
            dataset = data.info_audio_dataset.InfoAudioDataset.from_meta(path, return_info=return_info, **kwargs)
        else:
            raise ValueError(f"Dataset type is unsupported: {dataset_type}")

        loader = get_loader(
            dataset,
            num_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            collate_fn=dataset.collater if return_info else None,
            shuffle=shuffle,
        )
        dataloaders[split] = loader

    return dataloaders
