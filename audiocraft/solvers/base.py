# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
import typing as tp

import flashy
import omegaconf
import torch
from torch import nn

from .. import optim
from ..optim import fsdp
from ..utils import checkpoint
from ..utils.autocast import TorchAutocast
from ..utils.best_state import BestStateDictManager
from ..utils.deadlock import DeadlockDetect
from ..utils.profiler import Profiler
from ..utils.utils import copy_state, dict_from_config, model_hash, with_rank_rng


class StandardSolver(ABC, flashy.BaseSolver):
    """Standard solver for AudioCraft.

    The standard solver implements a base training loop with the following stages:
    train, valid, evaluate and generate that are expected to be all defined for
    solvers in AudioCraft. It also provides a nice default management of Dora history replay,
    checkpoint management across epoch, and logging configuration.

    AudioCraft solvers must inherit from the StandardSolver and define the methods
    associated to each stage as well as the show, build_model and build_dataloaders methods.
    """
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.logger.info(f"Instantiating solver {self.__class__.__name__} for XP {self.xp.sig}")
        self.logger.info(f"All XP logs are stored in {self.xp.folder}")
        self.cfg = cfg
        self.device = cfg.device
        self.model: nn.Module
        self._continue_best_source_keys = ['best_state', 'fsdp_best_state']
        self._fsdp_modules: tp.List[fsdp.FSDP] = []
        self._ema_sources: nn.ModuleDict = nn.ModuleDict()
        self.ema: tp.Optional[optim.ModuleDictEMA] = None
        self.dataloaders: tp.Dict[str, torch.utils.data.DataLoader] = dict()
        self._log_updates = self.cfg.logging.get('log_updates', 10)
        if self.cfg.logging.log_tensorboard:
            self.init_tensorboard(**self.cfg.get('tensorboard'))
        if self.cfg.logging.log_wandb and self:
            self.init_wandb(**self.cfg.get('wandb'))
        # keep a copy of the best performing state for stateful objects
        # used for evaluation and generation stages
        dtype_best: tp.Optional[torch.dtype] = None
        if self.cfg.fsdp.use:
            dtype_best = getattr(torch, self.cfg.fsdp.param_dtype)  # type: ignore
            assert isinstance(dtype_best, torch.dtype)
        elif self.cfg.autocast:
            dtype_best = getattr(torch, self.cfg.autocast_dtype)  # type: ignore
            assert isinstance(dtype_best, torch.dtype)
        self.best_state: BestStateDictManager = BestStateDictManager(dtype=dtype_best)
        # Hacky support for keeping a copy of the full best state in rank0.
        self.fsdp_best_state: tp.Dict[str, tp.Any] = {}
        self.register_stateful('best_state', 'fsdp_best_state')  # register best_state object to keep it in state_dict
        self._new_best_state: bool = False  # should save a new checkpoint
        # instantiate datasets and appropriate number of updates per epoch
        self.build_dataloaders()
        if self.cfg.execute_only is None:
            assert 'train' in self.dataloaders, "The train dataset split must be provided."
            assert 'valid' in self.dataloaders, "The valid dataset split must be provided."
        self.train_updates_per_epoch = len(self.dataloaders['train']) if 'train' in self.dataloaders else 0
        if self.cfg.optim.updates_per_epoch:
            self.train_updates_per_epoch = self.cfg.optim.updates_per_epoch
        self.total_updates = self.train_updates_per_epoch * self.cfg.optim.epochs
        # instantiate model & exponential moving average on the model
        self.build_model()
        self.logger.info("Model hash: %s", model_hash(self.model))
        assert 'model' in self.stateful.sources, \
            "Please register the model to stateful with self.register_stateful('model') in build_model."
        self.profiler = Profiler(self.model, **self.cfg.profiler)
        self.initialize_ema()
        self.register_stateful('ema')
        assert self.ema is None or 'ema' in self.stateful.sources, \
            "Please register the ema to stateful with self.register_stateful('ema') in build_model."
        self.deadlock_detect = DeadlockDetect(**self.cfg.deadlock)
        # basic statistics on the trained model
        model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        # one copy of grad, one copy of momentum, one copy of denominator and model weights.
        # and 4 bytes for each float!
        mem_usage = model_size * 4 * 4 / 1000
        self.logger.info("Model size: %.2f M params", model_size)
        self.logger.info("Base memory usage, with model, grad and optim: %.2f GB", mem_usage)

    @property
    def autocast(self):
        """Convenient autocast (or not) using the solver configuration."""
        return TorchAutocast(enabled=self.cfg.autocast, device_type=self.device, dtype=self.autocast_dtype)

    def _get_state_source(self, name) -> flashy.state.StateDictSource:
        # Internal utility to get a state source from the solver
        return self.stateful.sources[name]

    @property
    def best_metric_name(self) -> tp.Optional[str]:
        """Metric name used to identify the best state. This metric should be stored in the metrics
        used on the stage for best state identification (most likely, `valid`). If None, then
        no best state is saved.
        """
        return None

    def register_best_state(self, *args: str):
        """Register state sources in `BestStateDictManager` to keep their best states along with their
        latest states. The best state will be used at evaluation stages instead of the latest states.

        Shortcut around `BestStateDictManager.register` method. You can pass any number of
        attribute, included nested attributes and those will be included into the checkpoints
        and automatically restored when `BaseSolver.restore` is called.
        """
        for name in args:
            state_source = self._get_state_source(name)
            assert name in self.stateful.sources, "Registered states in best should be registered in stateful first!"
            self.best_state.register(name, state_source)

    def register_ema(self, *args: str):
        """Register state sources for exponential moving average.

        The registered sources are used to instantiate a ModuleDictEMA instance.
        The ModuleDictEMA keeps a `nn.ModuleDict` module that is updated when self.ema.step() is called
        and swapped with the original state sources with self.swap_ema_state() method.

        Usage:
            self.register_ema('model')
        """
        assert self.ema is None, "Cannot register state source to already instantiated EMA."
        for name in args:
            self._ema_sources[name] = getattr(self, name)

    def wrap_with_fsdp(self, model: torch.nn.Module, *args, **kwargs):
        model = fsdp.wrap_with_fsdp(self.cfg.fsdp, model, *args, **kwargs)
        if isinstance(model, fsdp.FSDP):
            self._fsdp_modules.append(model)
        return model

    def update_best_state_from_stage(self, stage_name: str = 'valid'):
        """Update latest best state based on pending metrics of a given stage. This method relies
        on the `BestStateDictManager.update` method to update the best state_dict with latest weights
        if the registered states happen to match to the best performing setup.
        """
        if self.best_metric_name is None:
            # when no best metric is defined, the last state is always the best
            self._new_best_state = True
            self.logger.info("Updating best state with current state.")
        else:
            assert stage_name in self._pending_metrics, f"Metrics for stage {stage_name} not found."
            assert self.best_metric_name in self._pending_metrics[stage_name], \
                f"Best metric not found in {stage_name} metrics. Cannot register best state"
            current_score = self._pending_metrics[stage_name][self.best_metric_name]
            all_best_metric_scores = [
                past_metrics[stage_name][self.best_metric_name]
                for past_metrics in self.history
            ]
            all_best_metric_scores.append(current_score)
            best_score = min(all_best_metric_scores)
            self._new_best_state = current_score == best_score
            if self._new_best_state:
                old_best = min(all_best_metric_scores[:-1] + [float('inf')])
                self.logger.info(
                    f"New best state with {self.best_metric_name}={current_score:.3f} (was {old_best:.3f})")

        if self._new_best_state:
            if self.cfg.fsdp.use:
                # this will give an empty state dict on all ranks but the rank 0
                # which will have a copy in memory of the full model.
                with fsdp.switch_to_full_state_dict(self._fsdp_modules):
                    for name in self.best_state.states.keys():
                        state_source = self._get_state_source(name)
                        self.best_state.update(name, state_source)
                    # we save to a different dict.
                    self.fsdp_best_state.update(self.best_state.state_dict())
                # We cannot efficiently load fsdp_best_state when using FSDP,
                # so we have do do a second pass, with the local shards.
            for name in self.best_state.states.keys():
                state_source = self._get_state_source(name)
                self.best_state.update(name, state_source)

    def _load_new_state_dict(self, state_dict: dict) -> dict:
        old_states = {}
        for name, new_state in state_dict.items():
            state_source = self._get_state_source(name)
            old_states[name] = copy_state(state_source.state_dict())
            state_source.load_state_dict(new_state)
        return old_states

    @contextmanager
    def swap_best_state(self):
        self.logger.debug(f"Swapping to best state for: {', '.join(self.best_state.state_dict().keys())}")
        old_states = self._load_new_state_dict(self.best_state.state_dict())
        try:
            yield
        finally:
            self.logger.debug("Swapping back from best to original state")
            for name, old_state in old_states.items():
                state_source = self._get_state_source(name)
                state_source.load_state_dict(old_state)

    @contextmanager
    def swap_ema_state(self):
        if self.ema is None:
            yield
        else:
            ema_state_dict = self.ema.state_dict()['state']
            self.logger.debug(f"Swapping to EMA state for: {', '.join(ema_state_dict.keys())}")
            old_states = self._load_new_state_dict(ema_state_dict)
            try:
                yield
            finally:
                self.logger.debug("Swapping back from EMA state to original state")
                for name, old_state in old_states.items():
                    state_source = self._get_state_source(name)
                    state_source.load_state_dict(old_state)

    @property
    def is_training(self):
        return self.current_stage == 'train'

    def log_model_summary(self, model: nn.Module):
        """Log model summary, architecture and size of the model."""
        self.logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
        self.logger.info("Size: %.1f MB", mb)

    @abstractmethod
    def build_model(self):
        """Method to implement to initialize model."""
        ...

    def initialize_ema(self):
        """Initialize exponential moving average with the registered sources.
        EMA object is created if the optim.ema.model.decay value is non-null.
        """
        from .builders import get_ema
        self.ema = get_ema(self._ema_sources, self.cfg.optim.ema)
        if self.ema is None:
            self.logger.info('No EMA on the model.')
        else:
            assert self.cfg.optim.ema.updates > 0
            self.logger.info(
                f'Initializing EMA on the model with decay = {self.ema.decay}'
                f' every {self.cfg.optim.ema.updates} updates'
            )

    @abstractmethod
    def build_dataloaders(self):
        """Method to implement to initialize dataloaders."""
        ...

    @abstractmethod
    def show(self):
        """Method to log any information without running the job."""
        ...

    @property
    def log_updates(self):
        # convenient access to log updates
        return self._log_updates

    def checkpoint_path(self, **kwargs):
        kwargs.setdefault('use_fsdp', self.cfg.fsdp.use)
        return self.folder / checkpoint.checkpoint_name(**kwargs)

    def epoch_checkpoint_path(self, epoch: int, **kwargs):
        kwargs.setdefault('use_fsdp', self.cfg.fsdp.use)
        return self.folder / checkpoint.checkpoint_name(str(epoch), **kwargs)

    def checkpoint_path_with_name(self, name: str, **kwargs):
        kwargs.setdefault('use_fsdp', self.cfg.fsdp.use)
        return self.folder / checkpoint.checkpoint_name(name=name, **kwargs)

    def save_checkpoints(self):
        """Save checkpoint, optionally keeping a copy for a given epoch."""
        is_sharded = self.cfg.fsdp.use
        if not flashy.distrib.is_rank_zero() and not is_sharded:
            return
        self.logger.info("Model hash: %s", model_hash(self.model))
        state = self.state_dict()
        epoch = self.epoch - 1  # pushing metrics will increase the epoch in Flashy, so we do -1 here

        # save minimal state_dict as new checkpoint every X epoch
        if self.cfg.checkpoint.save_every:
            if epoch % self.cfg.checkpoint.save_every == 0:
                minimal_state = state
                if self.cfg.checkpoint.keep_every_states is not None and len(self.cfg.checkpoint.keep_every_states) > 0:
                    minimal_state = {
                        name: source for name, source in state.items()
                        if name in self.cfg.checkpoint.keep_every_states
                    }
                epoch_checkpoint_path = self.epoch_checkpoint_path(epoch)
                checkpoint.save_checkpoint(minimal_state, epoch_checkpoint_path, is_sharded)

        # save checkpoint as latest checkpoint
        if self.cfg.checkpoint.save_last:
            last_checkpoint_path = self.checkpoint_path()
            checkpoint.save_checkpoint(state, last_checkpoint_path, is_sharded)

        # flush any stale checkpoint to reduce disk footprint
        checkpoint.flush_stale_checkpoints(self.checkpoint_path())

    def load_from_pretrained(self, name: str) -> dict:
        raise NotImplementedError("Solver does not provide a way to load pretrained models.")

    def load_checkpoints(self, load_best: bool = False, ignore_state_keys: tp.List[str] = []) -> tp.Optional[dict]:
        """Load last checkpoint or the one specified in continue_from.

        Args:
            load_best (bool): Whether to load from best state dict or not.
                Best state dict is always used when not loading the current xp.
            ignore_state_keys (list of str): List of sources to ignore when loading the state, e.g. `optimizer`.
        Returns:
            state (dict, optional): The loaded state dictionary.
        """
        # load checkpoints from xp folder or cfg.continue_from
        is_sharded = self.cfg.fsdp.use
        load_from_path: tp.Optional[Path] = None
        checkpoint_source: tp.Optional[checkpoint.CheckpointSource] = None

        if load_best:
            self.logger.info("Trying to load state_dict from best state.")

        state: tp.Optional[dict] = None
        rank0_checkpoint_path = self.checkpoint_path(use_fsdp=False)
        current_checkpoint_path = self.checkpoint_path()
        _pretrained_prefix = '//pretrained/'
        continue_pretrained = (self.cfg.continue_from or '').startswith(_pretrained_prefix)
        if rank0_checkpoint_path.exists():
            self.logger.info(f"Loading existing checkpoint: {current_checkpoint_path}")
            load_from_path = current_checkpoint_path
            checkpoint.check_sharded_checkpoint(current_checkpoint_path, rank0_checkpoint_path)
            checkpoint_source = checkpoint.CheckpointSource.CURRENT_XP
        elif self.cfg.continue_from and not continue_pretrained:
            self.logger.info(f"Continuing from provided checkpoint: {self.cfg.continue_from}")
            # we're always continuing from consolidated checkpoints: self.cfg.use_fsdp and not continue_best
            load_from_path = checkpoint.resolve_checkpoint_path(self.cfg.continue_from, use_fsdp=False)
            if load_from_path is None:
                self.logger.error('Could not resolve the continue_from checkpoint %s', self.cfg.continue_from)
                raise RuntimeError(f'Could not resolve continue_from checkpoint {self.cfg.continue_from}')
            checkpoint_source = checkpoint.CheckpointSource.OTHER

        if load_from_path is not None:
            state = checkpoint.load_checkpoint(load_from_path, is_sharded)
        elif continue_pretrained:
            self.logger.info("Loading a pretrained model. Ignoring 'load_best' and 'ignore_state_keys' params.")
            state = self.load_from_pretrained(self.cfg.continue_from[len(_pretrained_prefix):])
            checkpoint_source = checkpoint.CheckpointSource.PRETRAINED
            load_best = True

        # checkpoints are not from the current xp, we only retrieve the best state
        if checkpoint_source is not None and checkpoint_source != checkpoint.CheckpointSource.CURRENT_XP:
            assert state is not None
            self.logger.info("Checkpoint source is not the current xp: Load state_dict from best state.")
            load_best = True
            state = {key: state[key] for key in self._continue_best_source_keys if key in state}
            # loaded checkpoints are FSDP checkpoints: we're reading the best state
            # from FSDP and we drop the regular best_state
            if 'fsdp_best_state' in state and state['fsdp_best_state']:
                state.pop('best_state', None)
                self.logger.info("... Loaded checkpoint has FSDP best state")
            # FSDP is enabled in the solver, if the loaded checkpoints do not have FSDP support
            # then we're initializing FSDP best state with the regular best state
            elif self.cfg.fsdp.use:
                if 'fsdp_best_state' not in state or not state['fsdp_best_state']:
                    # we swap non-FSDP checkpoints best_state to FSDP-compatible best state
                    state['fsdp_best_state'] = state.pop('best_state')
                    self.logger.info("... Loaded checkpoint does not have FSDP best state. Use regular best state")

        if state is not None:
            if load_best:
                self.logger.info("Ignoring keys when loading best %r", ignore_state_keys)
                for key in set(ignore_state_keys):
                    if key in state:
                        state.pop(key)
                has_best_state = 'best_state' in state or 'fsdp_best_state' in state
                assert has_best_state, ("Trying to load best state but neither 'best_state'",
                                        " or 'fsdp_best_state' found in checkpoints.")
            self.load_state_dict(state)

        # for FSDP, let's make extra sure nothing bad happened with out of sync
        # checkpoints across workers.
        epoch = float(self.epoch)
        avg_epoch = flashy.distrib.average_metrics({'epoch': epoch})['epoch']
        if avg_epoch != epoch:
            raise RuntimeError(
                f"Inconsistent loading of checkpoints happened, our epoch is {epoch} "
                f"but average of epochs is {avg_epoch}, at least one gpu must have a "
                "different epoch number.")

        # on load_best, properly reinitialize state_dict, best states and ema
        # otherwise we load from the current xp and don't alter anything
        if load_best:
            self.logger.info("Loading state_dict from best state.")
            if not self.cfg.fsdp.use and self.fsdp_best_state:
                # loading from an FSDP checkpoint but with FSDP deactivated
                self.logger.info("... Loading from FSDP best state dict.")
                self.best_state.load_state_dict(self.fsdp_best_state)

            # if load_best, we permanently override the regular state_dict with the best state
            if self.cfg.fsdp.use:
                self.logger.info("FSDP is used, loading from FSDP best state.")
                with fsdp.switch_to_full_state_dict(self._fsdp_modules):
                    # this might be really fragile but okay for now.
                    self.load_state_dict(self.fsdp_best_state)
            else:
                # we permanently swap the stateful objects to their best state
                self._load_new_state_dict(self.best_state.state_dict())

            # the EMA modules should also be instantiated with best state.
            # the easiest way to do so is to reinitialize a new EMA with best state loaded.
            if self.ema is not None:
                self.logger.info("Re-initializing EMA from best state")
                self.initialize_ema()

            if self.cfg.fsdp.use:
                self.logger.info("Re-initializing best state after using FSDP best state.")
                for name in self.best_state.states.keys():
                    state_source = self._get_state_source(name)
                    self.best_state.update(name, state_source)

        return state

    def restore(self, load_best: bool = False, replay_metrics: bool = False,
                ignore_state_keys: tp.List[str] = []) -> bool:
        """Restore the status of a solver for a given xp.

        Args:
            load_best (bool): if `True`, load the best state from the checkpoint.
            replay_metrics (bool): if `True`, logs all the metrics from past epochs.
            ignore_state_keys (list of str): list of sources to ignore when loading the state, e.g. `optimizer`.
        """
        self.logger.info("Restoring weights and history.")
        restored_checkpoints = self.load_checkpoints(load_best, ignore_state_keys)

        self.logger.info("Model hash: %s", model_hash(self.model))

        if replay_metrics and len(self.history) > 0:
            self.logger.info("Replaying past metrics...")
            for epoch, stages in enumerate(self.history):
                for stage_name, metrics in stages.items():
                    # We manually log the metrics summary to the result logger
                    # as we don't want to add them to the pending metrics
                    self.result_logger._log_summary(stage_name, metrics, step=epoch + 1, step_name='epoch',
                                                    formatter=self.get_formatter(stage_name))
        return restored_checkpoints is not None

    def commit(self, save_checkpoints: bool = True):
        """Commit metrics to dora and save checkpoints at the end of an epoch."""
        # we override commit to introduce more complex checkpoint saving behaviors
        self.history.append(self._pending_metrics)  # This will increase self.epoch
        if save_checkpoints:
            self.save_checkpoints()
        self._start_epoch()
        if flashy.distrib.is_rank_zero():
            self.xp.link.update_history(self.history)

    def run_epoch(self):
        """Run a single epoch with all stages.

        Metrics for a given stage are stored in _pending_metrics and committed by the solver afterwards.
        Children solvers can extend this method with custom behavior, e.g.:

            def run_epoch(self):
                ... # custom code
                super().run_epoch()
                ... # custom code
        """
        self.run_stage('train', self.train)
        with torch.no_grad():
            with self.swap_ema_state():
                self.run_stage('valid', self.valid)
                # the best state is updated with EMA states if available
                self.update_best_state_from_stage('valid')
            with self.swap_best_state():
                if self.should_run_stage('evaluate'):
                    self.run_stage('evaluate', self.evaluate)
                if self.should_run_stage('generate'):
                    self.run_stage('generate', with_rank_rng()(self.generate))

    def run(self):
        """Training loop."""
        assert len(self.state_dict()) > 0
        self.restore(replay_metrics=True)  # load checkpoint and replay history
        self.log_hyperparams(dict_from_config(self.cfg))
        for epoch in range(self.epoch, self.cfg.optim.epochs + 1):
            if self.should_stop_training():
                return
            self.run_epoch()
            # Commit will send the metrics to Dora and save checkpoints by default.
            self.commit()

    def should_stop_training(self) -> bool:
        """Check whether we should stop training or not."""
        return self.epoch > self.cfg.optim.epochs

    def should_run_stage(self, stage_name) -> bool:
        """Check whether we want to run the specified stages."""
        stage_every = self.cfg[stage_name].get('every', None)
        is_last_epoch = self.epoch == self.cfg.optim.epochs
        is_epoch_every = (stage_every and self.epoch % stage_every == 0)
        return is_last_epoch or is_epoch_every

    @abstractmethod
    def run_step(self, idx: int, batch: tp.Any, metrics: dict):
        """Perform one training or valid step on a given batch."""
        ...

    def common_train_valid(self, dataset_split: str, **kwargs: tp.Any):
        """Common logic for train and valid stages."""
        self.model.train(self.is_training)

        loader = self.dataloaders[dataset_split]
        # get a different order for distributed training, otherwise this will get ignored
        if flashy.distrib.world_size() > 1 \
           and isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
            loader.sampler.set_epoch(self.epoch)
        updates_per_epoch = self.train_updates_per_epoch if self.is_training else len(loader)
        if self.cfg.benchmark_no_load:
            self.logger.warning("Fake loading for benchmarking: re-using first batch")
            batch = next(iter(loader))
            loader = [batch] * updates_per_epoch  # type: ignore
        lp = self.log_progress(self.current_stage, loader, total=updates_per_epoch, updates=self.log_updates)
        average = flashy.averager()  # epoch wise average
        instant_average = flashy.averager()  # average between two logging
        metrics: dict = {}

        with self.profiler, self.deadlock_detect:  # profiler will only run for the first 20 updates.
            for idx, batch in enumerate(lp):
                self.deadlock_detect.update('batch')
                if idx >= updates_per_epoch:
                    break
                metrics = {}
                metrics = self.run_step(idx, batch, metrics)
                self.deadlock_detect.update('step')
                # run EMA step
                if self.ema is not None and self.is_training and (idx + 1) % self.cfg.optim.ema.updates == 0:
                    self.logger.debug("EMA model step")
                    self.ema.step()
                self.deadlock_detect.update('ema')
                self.profiler.step()
                instant_metrics = instant_average(metrics)
                if lp.update(**instant_metrics):
                    instant_average = flashy.averager()  # reset averager between two logging
                metrics = average(metrics)  # epoch wise average
                self.deadlock_detect.update('end_batch')

        metrics = flashy.distrib.average_metrics(metrics, updates_per_epoch)
        return metrics

    def train(self):
        """Train stage."""
        return self.common_train_valid('train')

    def valid(self):
        """Valid stage."""
        return self.common_train_valid('valid')

    @abstractmethod
    def evaluate(self):
        """Evaluate stage."""
        ...

    @abstractmethod
    def generate(self):
        """Generate stage."""
        ...

    def run_one_stage(self, stage_name: str):
        """Run only the specified stage.
        This method is useful to only generate samples from a trained experiment
        or rerun the validation or evaluation stages.
        """
        fn = {
            'generate': with_rank_rng()(self.generate),
            'evaluate': self.evaluate,
            'valid': self.valid,
        }
        if stage_name not in fn:
            raise ValueError(f'Trying to run stage {stage_name} is not supported.')
        assert len(self.state_dict()) > 0
        self._start_epoch()
        with torch.no_grad(), self.swap_best_state():
            self.run_stage(stage_name, fn[stage_name])
        if not self.cfg.execute_inplace:
            self.commit(save_checkpoints=False)

    @staticmethod
    def get_eval_solver_from_sig(sig: str, dtype: tp.Optional[str] = None,
                                 device: tp.Optional[str] = None, autocast: bool = True,
                                 batch_size: tp.Optional[int] = None,
                                 override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
                                 **kwargs):
        """Mostly a convenience function around audiocraft.train.get_solver_from_sig,
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
