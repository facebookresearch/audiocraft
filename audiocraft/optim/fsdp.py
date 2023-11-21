# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper around FSDP for more convenient use in the training loops.
"""

from contextlib import contextmanager
import typing as tp
import dora
import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision, ShardingStrategy, FullStateDictConfig, StateDictType)
from torch.distributed._shard.sharded_tensor.api import ShardedTensor


def is_fsdp_used() -> bool:
    """Return whether we are using FSDP."""
    # A bit of a hack but should work from anywhere.
    if dora.is_xp():
        cfg = dora.get_xp().cfg
        if hasattr(cfg, 'fsdp'):
            return cfg.fsdp.use
    return False


def is_sharded_tensor(x: tp.Any) -> bool:
    return isinstance(x, ShardedTensor)


@contextmanager
def switch_to_full_state_dict(models: tp.List[FSDP]):
    # Another bug in FSDP makes it that we cannot use the `state_dict_type` API,
    # so let's do thing manually.
    for model in models:
        FSDP.set_state_dict_type(  # type: ignore
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True))
    try:
        yield
    finally:
        for model in models:
            FSDP.set_state_dict_type(model, StateDictType.LOCAL_STATE_DICT)  # type: ignore


def wrap_with_fsdp(cfg, model: torch.nn.Module,
                   block_classes: tp.Optional[tp.Set[tp.Type]] = None) -> FSDP:
    """Wraps a model with FSDP."""
    # Some of the typing is disabled until this gets integrated
    # into the stable version of PyTorch.
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy  # type: ignore

    # we import this here to prevent circular import.
    from ..modules.transformer import StreamingTransformerLayer
    from ..modules.conditioners import ConditioningProvider

    _fix_post_backward_hook()

    assert cfg.use
    sharding_strategy_dict = {
        "no_shard": ShardingStrategy.NO_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "full_shard": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[cfg.param_dtype],
        reduce_dtype=dtype_dict[cfg.reduce_dtype],
        buffer_dtype=dtype_dict[cfg.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[cfg.sharding_strategy]
    # The following is going to require being a bit smart
    # when doing LM, because this would flush the weights for every time step
    # during generation. One possiblity is to use hybrid sharding:
    # See: https://pytorch.org/docs/master/fsdp.html#torch.distributed.fsdp.ShardingStrategy
    assert sharding_strategy_config != ShardingStrategy.FULL_SHARD, \
        "Not supported at the moment, requires a bit more work."

    local_rank = dora.distrib.get_distrib_spec().local_rank
    assert local_rank < torch.cuda.device_count(), "Please upgrade Dora!"

    auto_wrap_policy = None
    if block_classes is None:
        block_classes = {StreamingTransformerLayer, ConditioningProvider}
    if cfg.per_block:
        auto_wrap_policy = ModuleWrapPolicy(block_classes)
    wrapped = _FSDPFixStateDict(
        model,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=auto_wrap_policy,
    )  # type: ignore
    FSDP.set_state_dict_type(wrapped, StateDictType.LOCAL_STATE_DICT)  # type: ignore

    # Let the wrapped model know about the wrapping!
    # We use __dict__ to avoid it going into the state dict.
    # This is a bit dirty, but needed during generation, as otherwise
    # the wrapped model would call itself and bypass FSDP.
    for module in FSDP.fsdp_modules(wrapped):
        original = module._fsdp_wrapped_module
        original.__dict__['_fsdp'] = module
    return wrapped


def purge_fsdp(model: FSDP):
    """Purge the FSDP cached shard inside the model. This should
    allow setting the best state or switching to the EMA.
    """
    from torch.distributed.fsdp._runtime_utils import _reshard  # type: ignore
    for module in FSDP.fsdp_modules(model):
        handles = module._handles
        if not handles:
            continue
        handle = handles[0]
        unsharded_flat_param = handle._get_padded_unsharded_flat_param()
        storage_size: int = unsharded_flat_param._typed_storage()._size()  # type: ignore
        if storage_size == 0:
            continue
        true_list = [True for h in handles]
        _reshard(module, handles, true_list)


class _FSDPFixStateDict(FSDP):
    @staticmethod
    def _name_without_fsdp_prefix(name: str) -> str:
        from torch.distributed.fsdp._common_utils import FSDP_WRAPPED_MODULE  # type: ignore
        parts = name.split('.')
        new_parts = [part for part in parts if part != FSDP_WRAPPED_MODULE]
        return '.'.join(new_parts)

    def state_dict(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:  # type: ignore
        state = dict(super().state_dict(*args, **kwargs))
        for key, value in list(state.items()):
            if is_sharded_tensor(value):
                del state[key]
        return state

    def load_state_dict(self, state: tp.Dict[str, tp.Any]):  # type: ignore
        if self._state_dict_type is StateDictType.FULL_STATE_DICT:
            super().load_state_dict(state)
            purge_fsdp(self)
            return
        # Fix FSDP load state dict in all situation.
        # Use this only with LOCAL_STATE_DICT !!!
        current_state = dict(super().state_dict())
        for key, value in state.items():
            key = _FSDPFixStateDict._name_without_fsdp_prefix(key)
            if key not in current_state:
                # Emulate strict loading manually.
                raise RuntimeError(f"Unknown state key {key}")
            current_state[key].copy_(value)

        # Purging cached weights from previous forward.
        purge_fsdp(self)


_hook_fixed = False


def _fix_post_backward_hook():
    global _hook_fixed
    if _hook_fixed:
        return
    _hook_fixed = True

    from torch.distributed.fsdp import _runtime_utils
    from torch.distributed.fsdp._common_utils import TrainingState, HandleTrainingState
    old_hook = _runtime_utils._post_backward_hook

    def _post_backward_hook(state, handle, *args, **kwargs):
        checkpointed = getattr(state._fsdp_wrapped_module, '_audiocraft_checkpointed', False)
        if checkpointed:
            # there will be one more forward in the backward with checkpointing and that will
            # massively confuse FSDP, so we have to make it think everything
            # is going according to the plan.
            state.training_state = TrainingState.FORWARD_BACKWARD
            handle._training_state = HandleTrainingState.BACKWARD_PRE
        old_hook(state, handle, *args, **kwargs)

    _runtime_utils._post_backward_hook = _post_backward_hook
