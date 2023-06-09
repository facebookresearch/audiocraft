# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions to load from the checkpoints.
Each checkpoint is a torch.saved dict with the following keys:
- 'xp.cfg': the hydra config as dumped during training. This should be used
    to rebuild the object using the audiocraft.models.builders functions,
- 'model_best_state': a readily loadable best state for the model, including
    the conditioner. The model obtained from `xp.cfg` should be compatible
    with this state dict. In the case of a LM, the encodec model would not be
    bundled along but instead provided separately.

Those functions also support loading from a remote location with the Torch Hub API.
They also support overriding some parameters, in particular the device and dtype
of the returned model.
"""

from pathlib import Path
import typing as tp

from omegaconf import OmegaConf
import torch

from . import builders


def _get_state_dict(file_or_url: tp.Union[Path, str], device='cpu'):
    # Return the state dict either from a file or url
    file_or_url = str(file_or_url)
    assert isinstance(file_or_url, str)
    if file_or_url.startswith('https://'):
        return torch.hub.load_state_dict_from_url(file_or_url, map_location=device, check_hash=True)
    else:
        return torch.load(file_or_url, device)


def load_compression_model(file_or_url: tp.Union[Path, str], device='cpu'):
    pkg = _get_state_dict(file_or_url)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    return model


def load_lm_model(file_or_url: tp.Union[Path, str], device='cpu'):
    pkg = _get_state_dict(file_or_url)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.transformer_lm.memory_efficient = False
        cfg.transformer_lm.custom = True
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model
