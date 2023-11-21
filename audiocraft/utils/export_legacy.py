# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Legacy functions used at the time of the first release, kept for referencd.
"""

from pathlib import Path
import typing as tp

from omegaconf import OmegaConf, DictConfig
import torch

from audiocraft import __version__


def _clean_lm_cfg(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    # This used to be set automatically in the LM solver, need a more robust solution
    # for the future.
    cfg['transformer_lm']['card'] = 2048
    n_q = 4
    stereo_cfg = getattr(cfg, 'interleave_stereo_codebooks', None)
    if stereo_cfg is not None and stereo_cfg.use:
        if 'downsample' in stereo_cfg:
            del stereo_cfg['downsample']
        n_q = 8
    cfg['transformer_lm']['n_q'] = n_q
    # Experimental params no longer supported.
    bad_params = ['spectral_norm_attn_iters', 'spectral_norm_ff_iters',
                  'residual_balancer_attn', 'residual_balancer_ff', 'layer_drop']
    for name in bad_params:
        del cfg['transformer_lm'][name]
    OmegaConf.set_struct(cfg, True)
    return cfg


def export_encodec(checkpoint_path: tp.Union[Path, str], out_file: tp.Union[Path, str]):
    pkg = torch.load(checkpoint_path, 'cpu')
    new_pkg = {
        'best_state': pkg['ema']['state']['model'],
        'xp.cfg': OmegaConf.to_yaml(pkg['xp.cfg']),
        # The following params were NOT exported for the first release of MusicGen.
        'version': __version__,
        'exported': True,
    }
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    torch.save(new_pkg, out_file)
    return out_file


def export_lm(checkpoint_path: tp.Union[Path, str], out_file: tp.Union[Path, str]):
    pkg = torch.load(checkpoint_path, 'cpu')
    if pkg['fsdp_best_state']:
        best_state = pkg['fsdp_best_state']['model']
    else:
        best_state = pkg['best_state']['model']
    new_pkg = {
        'best_state': best_state,
        'xp.cfg': OmegaConf.to_yaml(_clean_lm_cfg(pkg['xp.cfg'])),
        # The following params were NOT exported for the first release of MusicGen.
        'version': __version__,
        'exported': True,
    }
    Path(out_file).parent.mkdir(exist_ok=True, parents=True)
    torch.save(new_pkg, out_file)
    return out_file
