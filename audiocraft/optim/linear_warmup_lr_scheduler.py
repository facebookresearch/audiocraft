# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupLRScheduler(_LRScheduler):
    """Inverse square root LR scheduler.

    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        warmup_init_lr (tp.Optional[float]): Initial learning rate
            during warmup phase. When not set, use the provided learning rate.
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, warmup_init_lr: tp.Optional[float] = 0):
        self.warmup_steps = warmup_steps
        self.warmup_init_lr = warmup_init_lr
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            warmup_init_lr = self.warmup_init_lr or 0
            lr_step = (lr - warmup_init_lr) / self.warmup_steps
            lr = warmup_init_lr + step * lr_step
        return lr

    def get_lr(self):
        return [self._get_sched_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]
