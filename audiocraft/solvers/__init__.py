# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Solvers. A Solver is a training recipe, combining the dataloaders, models,
optimizer, losses etc into a single convenient object.
"""

from .audiogen import AudioGenSolver
from .builders import get_solver
from .base import StandardSolver
from .compression import CompressionSolver
from .musicgen import MusicGenSolver
from .diffusion import DiffusionSolver

__all__ = [
    'AudioGenSolver',
    'CompressionSolver',
    'DiffusionSolver',
    'MusicGenSolver',
    'StandardSolver',
    'get_solver',
]
