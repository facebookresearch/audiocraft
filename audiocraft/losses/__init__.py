# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Loss related classes and functions. In particular the loss balancer from
EnCodec, and the usual spectral losses."""

# flake8: noqa
from .balancer import Balancer
from .sisnr import SISNR
from .stftloss import (
    LogSTFTMagnitudeLoss,
    MRSTFTLoss,
    SpectralConvergenceLoss,
    STFTLoss
)
from .specloss import (
    MelSpectrogramL1Loss,
    MultiScaleMelSpectrogramLoss,
)
