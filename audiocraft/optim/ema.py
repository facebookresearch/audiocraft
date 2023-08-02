# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ModelEMA implementation is taken from
# https://github.com/facebookresearch/demucs

from collections import defaultdict
import typing as tp

import torch
import torch.nn as nn


def _get_all_non_persistent_buffers_set(module: nn.Module, root: str = "") -> set:
    names: set = set()
    for (name, sub_module) in module.named_modules():
        if name == '':
            buffer_names = module._non_persistent_buffers_set
            buffer_names = {f"{root}.{buff_name}" if len(root) > 0 else buff_name
                            for buff_name in buffer_names}
            names.update(buffer_names)
        else:
            sub_name = f"{root}.{name}" if len(root) > 0 else name
            sub_buffer_names = _get_all_non_persistent_buffers_set(sub_module, sub_name)
            names.update(sub_buffer_names)
    return names


def _get_named_tensors(module: nn.Module):
    non_persistent_buffers_set = _get_all_non_persistent_buffers_set(module)
    named_buffers = [(name, buffer) for (name, buffer) in module.named_buffers()
                     if name not in non_persistent_buffers_set]
    named_parameters = list(module.named_parameters())
    return named_parameters + named_buffers


class ModuleDictEMA:
    """Exponential Moving Average over a nn.ModuleDict.

    You can switch to the EMA weights temporarily.
    """
    def __init__(self, module_dict: nn.ModuleDict, decay: float = 0.999,
                 unbias: bool = True, device: tp.Union[torch.device, str] = 'cpu'):
        self.decay = decay
        self.module_dict = module_dict
        self.state: dict = defaultdict(dict)
        self.count = 0
        self.device = device
        self.unbias = unbias
        self._init()

    def _init(self):
        for module_name, module in self.module_dict.items():
            for key, val in _get_named_tensors(module):
                if not val.is_floating_point():
                    continue
                device = self.device or val.device
                if key not in self.state[module_name]:
                    self.state[module_name][key] = val.detach().to(device, copy=True)

    def step(self):
        if self.unbias:
            self.count = self.count * self.decay + 1
            w = 1 / self.count
        else:
            w = 1 - self.decay
        for module_name, module in self.module_dict.items():
            for key, val in _get_named_tensors(module):
                if not val.is_floating_point():
                    continue
                device = self.device or val.device
                self.state[module_name][key].mul_(1 - w)
                self.state[module_name][key].add_(val.detach().to(device), alpha=w)

    def state_dict(self):
        return {'state': self.state, 'count': self.count}

    def load_state_dict(self, state):
        self.count = state['count']
        for module_name, module in state['state'].items():
            for key, val in module.items():
                self.state[module_name][key].copy_(val)
