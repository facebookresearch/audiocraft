# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Optimization stuff. In particular, optimizers (DAdaptAdam), schedulers
and Exponential Moving Average.
"""

# flake8: noqa
from .cosine_lr_scheduler import CosineLRScheduler
from .dadam import DAdaptAdam
from .inverse_sqrt_lr_scheduler import InverseSquareRootLRScheduler
from .linear_warmup_lr_scheduler import LinearWarmupLRScheduler
from .polynomial_decay_lr_scheduler import PolynomialDecayLRScheduler
from .ema import ModuleDictEMA
