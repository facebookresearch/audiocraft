# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import logging
import typing as tp

import flashy
import torch

from ..optim import ModuleDictEMA
from .utils import copy_state


logger = logging.getLogger(__name__)


class BestStateDictManager(flashy.state.StateDictSource):
    """BestStateDictManager maintains a copy of best state_dict() for registered sources.

    BestStateDictManager has two main attributes:
        states (dict): State dict of the registered StateDictSource.
        param_ids (dict): Dict of parameter ids for registered states from ModuleDictEMA and other sources.

    When registering new sources, the BestStateDictManager will ensure two conflicting sources between
    ModuleDictEMA and original modules are not both registered as it would otherwise create ambiguity about
    what to consider for best state.

    Args:
        device (torch.device or str): Device on which we keep the copy.
        dtype (torch.dtype): Data type for the state parameters.
    """
    def __init__(self, device: tp.Union[torch.device, str] = 'cpu',
                 dtype: tp.Optional[torch.dtype] = None):
        self.device = device
        self.states: dict = {}
        self.param_ids: dict = defaultdict(dict)
        self.dtype = dtype

    def _get_parameter_ids(self, state_dict):
        return {id(p): name for name, p in state_dict.items() if isinstance(p, torch.Tensor)}

    def _validate_no_parameter_ids_overlap(self, name: str, param_ids: dict):
        for registered_name, registered_param_ids in self.param_ids.items():
            if registered_name != name:
                overlap = set.intersection(registered_param_ids.keys(), param_ids.keys())
                assert len(overlap) == 0, f"Found {len(overlap)} / {len(param_ids.keys())} overlapping parameters"
                f" in {name} and already registered {registered_name}: {' '.join(overlap)}"

    def update(self, name: str, source: flashy.state.StateDictSource):
        if name not in self.states:
            raise ValueError(f"{name} missing from registered states.")
        self.states[name] = copy_state(source.state_dict(), device=self.device, dtype=self.dtype)

    def register(self, name: str, source: flashy.state.StateDictSource):
        if name in self.states:
            raise ValueError(f"{name} already present in states.")
        # Registering parameter ids for EMA and non-EMA states allows us to check that
        # there is no overlap that would create ambiguity about how to handle the best state
        param_ids = self._get_parameter_ids(source.state_dict())
        if isinstance(source, ModuleDictEMA):
            logger.debug(f"Registering to best state: ModuleDictEMA '{name}' with {len(param_ids)} params")
            self._validate_no_parameter_ids_overlap(name, param_ids)
            self.param_ids[name] = param_ids
        else:
            logger.debug(f"Registering to best state: StateDictSource '{name}' with {len(param_ids)} params")
            self._validate_no_parameter_ids_overlap('base', param_ids)
            self.param_ids['base'].update(param_ids)
        # Register state
        self.states[name] = copy_state(source.state_dict(), device=self.device, dtype=self.dtype)

    def state_dict(self) -> flashy.state.StateDict:
        return self.states

    def load_state_dict(self, state: flashy.state.StateDict):
        for name, sub_state in state.items():
            for k, v in sub_state.items():
                self.states[name][k].copy_(v)
