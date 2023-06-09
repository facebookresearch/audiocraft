# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility to export a training checkpoint to a lightweight release checkpoint.
"""

from pathlib import Path
import typing as tp

from omegaconf import OmegaConf, DictConfig
import torch


def _clean_lm_cfg(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    # This used to be set automatically in the LM solver, need a more robust solution
    # for the future.
    cfg['transformer_lm']['card'] = 2048
    cfg['transformer_lm']['n_q'] = 4
    # Experimental params no longer supported.
    bad_params = ['spectral_norm_attn_iters', 'spectral_norm_ff_iters',
                  'residual_balancer_attn', 'residual_balancer_ff', 'layer_drop']
    for name in bad_params:
        del cfg['transformer_lm'][name]
    OmegaConf.set_struct(cfg, True)
    return cfg


def export_encodec(checkpoint_path: tp.Union[Path, str], out_folder: tp.Union[Path, str]):
    sig = Path(checkpoint_path).parent.name
    assert len(sig) == 8, "Not a valid Dora signature"
    pkg = torch.load(checkpoint_path, 'cpu')
    new_pkg = {
        'best_state': pkg['ema']['state']['model'],
        'xp.cfg': OmegaConf.to_yaml(pkg['xp.cfg']),
    }
    out_file = Path(out_folder) / f'{sig}.th'
    torch.save(new_pkg, out_file)
    return out_file


def export_lm(checkpoint_path: tp.Union[Path, str], out_folder: tp.Union[Path, str]):
    sig = Path(checkpoint_path).parent.name
    assert len(sig) == 8, "Not a valid Dora signature"
    pkg = torch.load(checkpoint_path, 'cpu')
    new_pkg = {
        'best_state': pkg['fsdp_best_state']['model'],
        'xp.cfg': OmegaConf.to_yaml(_clean_lm_cfg(pkg['xp.cfg']))
    }
    out_file = Path(out_folder) / f'{sig}.th'
    torch.save(new_pkg, out_file)
    return out_file
