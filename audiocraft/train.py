# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Entry point for dora to launch solvers for running training loops.
See more info on how to use dora: https://github.com/facebookresearch/dora
"""

import logging
import multiprocessing
import os
from pathlib import Path
import sys
import typing as tp

from dora import git_save, hydra_main, XP
import flashy
import hydra
import omegaconf

from .environment import AudioCraftEnvironment
from .utils.cluster import get_slurm_parameters

logger = logging.getLogger(__name__)


def resolve_config_dset_paths(cfg):
    """Enable Dora to load manifest from git clone repository."""
    # manifest files for the different splits
    for key, value in cfg.datasource.items():
        if isinstance(value, str):
            cfg.datasource[key] = git_save.to_absolute_path(value)


def get_solver(cfg):
    from . import solvers
    # Convert batch size to batch size for each GPU
    assert cfg.dataset.batch_size % flashy.distrib.world_size() == 0
    cfg.dataset.batch_size //= flashy.distrib.world_size()
    for split in ['train', 'valid', 'evaluate', 'generate']:
        if hasattr(cfg.dataset, split) and hasattr(cfg.dataset[split], 'batch_size'):
            assert cfg.dataset[split].batch_size % flashy.distrib.world_size() == 0
            cfg.dataset[split].batch_size //= flashy.distrib.world_size()
    resolve_config_dset_paths(cfg)
    solver = solvers.get_solver(cfg)
    return solver


def get_solver_from_xp(xp: XP, override_cfg: tp.Optional[tp.Union[dict, omegaconf.DictConfig]] = None,
                       restore: bool = True, load_best: bool = True,
                       ignore_state_keys: tp.List[str] = [], disable_fsdp: bool = True):
    """Given a XP, return the Solver object.

    Args:
        xp (XP): Dora experiment for which to retrieve the solver.
        override_cfg (dict or None): If not None, should be a dict used to
            override some values in the config of `xp`. This will not impact
            the XP signature or folder. The format is different
            than the one used in Dora grids, nested keys should actually be nested dicts,
            not flattened, e.g. `{'optim': {'batch_size': 32}}`.
        restore (bool): If `True` (the default), restore state from the last checkpoint.
        load_best (bool): If `True` (the default), load the best state from the checkpoint.
        ignore_state_keys (list[str]): List of sources to ignore when loading the state, e.g. `optimizer`.
        disable_fsdp (bool): if True, disables FSDP entirely. This will
            also automatically skip loading the EMA. For solver specific
            state sources, like the optimizer, you might want to
            use along `ignore_state_keys=['optimizer']`. Must be used with `load_best=True`.
    """
    logger.info(f"Loading solver from XP {xp.sig}. "
                f"Overrides used: {xp.argv}")
    cfg = xp.cfg
    if override_cfg is not None:
        cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.DictConfig(override_cfg))
    if disable_fsdp and cfg.fsdp.use:
        cfg.fsdp.use = False
        assert load_best is True
        # ignoring some keys that were FSDP sharded like model, ema, and best_state.
        # fsdp_best_state will be used in that case. When using a specific solver,
        # one is responsible for adding the relevant keys, e.g. 'optimizer'.
        # We could make something to automatically register those inside the solver, but that
        # seem overkill at this point.
        ignore_state_keys = ignore_state_keys + ['model', 'ema', 'best_state']

    try:
        with xp.enter():
            solver = get_solver(cfg)
            if restore:
                solver.restore(load_best=load_best, ignore_state_keys=ignore_state_keys)
        return solver
    finally:
        hydra.core.global_hydra.GlobalHydra.instance().clear()


def get_solver_from_sig(sig: str, *args, **kwargs):
    """Return Solver object from Dora signature, i.e. to play with it from a notebook.
    See `get_solver_from_xp` for more information.
    """
    xp = main.get_xp_from_sig(sig)
    return get_solver_from_xp(xp, *args, **kwargs)


def init_seed_and_system(cfg):
    import numpy as np
    import torch
    import random
    from audiocraft.modules.transformer import set_efficient_attention_backend

    multiprocessing.set_start_method(cfg.mp_start_method)
    logger.debug('Setting mp start method to %s', cfg.mp_start_method)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # torch also initialize cuda seed if available
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads)
    os.environ['MKL_NUM_THREADS'] = str(cfg.num_threads)
    os.environ['OMP_NUM_THREADS'] = str(cfg.num_threads)
    logger.debug('Setting num threads to %d', cfg.num_threads)
    set_efficient_attention_backend(cfg.efficient_attention_backend)
    logger.debug('Setting efficient attention backend to %s', cfg.efficient_attention_backend)
    if 'SLURM_JOB_ID' in os.environ:
        tmpdir = Path('/scratch/slurm_tmpdir/' + os.environ['SLURM_JOB_ID'])
        if tmpdir.exists():
            logger.info("Changing tmpdir to %s", tmpdir)
            os.environ['TMPDIR'] = str(tmpdir)


@hydra_main(config_path='../config', config_name='config', version_base='1.1')
def main(cfg):
    init_seed_and_system(cfg)

    # Setup logging both to XP specific folder, and to stderr.
    log_name = '%s.log.{rank}' % cfg.execute_only if cfg.execute_only else 'solver.log.{rank}'
    flashy.setup_logging(level=str(cfg.logging.level).upper(), log_name=log_name)
    # Initialize distributed training, no need to specify anything when using Dora.
    flashy.distrib.init()
    solver = get_solver(cfg)
    if cfg.show:
        solver.show()
        return

    if cfg.execute_only:
        assert cfg.execute_inplace or cfg.continue_from is not None, \
            "Please explicitly specify the checkpoint to continue from with continue_from=<sig_or_path> " + \
            "when running with execute_only or set execute_inplace to True."
        solver.restore(replay_metrics=False)  # load checkpoint
        solver.run_one_stage(cfg.execute_only)
        return

    return solver.run()


main.dora.dir = AudioCraftEnvironment.get_dora_dir()
main._base_cfg.slurm = get_slurm_parameters(main._base_cfg.slurm)

if main.dora.shared is not None and not os.access(main.dora.shared, os.R_OK):
    print("No read permission on dora.shared folder, ignoring it.", file=sys.stderr)
    main.dora.shared = None

if __name__ == '__main__':
    main()
