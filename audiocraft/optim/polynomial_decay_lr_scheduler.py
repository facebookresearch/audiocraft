# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLRScheduler(_LRScheduler):
    """Polynomial decay LR scheduler.

    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.
        end_lr (float): Final learning rate to achieve over total number of steps.
        zero_lr_warmup_steps (int): Number of steps with a learning rate of value 0.
        power (float): Decay exponent.
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int,
                 end_lr: float = 0., zero_lr_warmup_steps: int = 0, power: float = 1.):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.zero_lr_warmup_steps = zero_lr_warmup_steps
        self.power = power
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if self.zero_lr_warmup_steps > 0 and step <= self.zero_lr_warmup_steps:
            lr = 0
        elif self.warmup_steps > 0 and step <= self.warmup_steps + self.zero_lr_warmup_steps:
            lr_ratio = (step - self.zero_lr_warmup_steps) / float(self.warmup_steps)
            lr = lr_ratio * lr
        elif step >= self.total_steps:
            lr = self.end_lr
        else:
            total_warmup_steps = self.warmup_steps + self.zero_lr_warmup_steps
            lr_range = lr - self.end_lr
            pct_remaining = 1 - (step - total_warmup_steps) / (self.total_steps - total_warmup_steps)
            lr = lr_range * pct_remaining ** self.power + self.end_lr
        return lr

    def get_lr(self):
        return [self._get_sched_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]
