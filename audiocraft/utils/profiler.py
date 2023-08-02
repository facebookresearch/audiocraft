# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import dora
import torch


logger = logging.getLogger(__name__)


class Profiler:
    """Context manager wrapper for xformers profiler.
    """
    def __init__(self, module: torch.nn.Module, enabled: bool = False):
        self.profiler: tp.Optional[tp.Any] = None
        if enabled:
            from xformers.profiler import profile
            output_dir = dora.get_xp().folder / 'profiler_data'
            logger.info("Profiling activated, results with be saved to %s", output_dir)
            self.profiler = profile(output_dir=output_dir, module=module)

    def step(self):
        if self.profiler is not None:
            self.profiler.step()  # type: ignore

    def __enter__(self):
        if self.profiler is not None:
            return self.profiler.__enter__()  # type: ignore

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.profiler is not None:
            return self.profiler.__exit__(exc_type, exc_value, exc_tb)  # type: ignore
