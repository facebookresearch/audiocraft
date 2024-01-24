# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Models for EnCodec, AudioGen, MusicGen, as well as the generic LMModel.
"""
# flake8: noqa
from . import builders, loaders
from .encodec import (
    CompressionModel, EncodecModel, DAC,
    HFEncodecModel, HFEncodecCompressionModel)
from .audiogen import AudioGen
from .lm import LMModel
from .lm_magnet import MagnetLMModel
from .multibanddiffusion import MultiBandDiffusion
from .musicgen import MusicGen
from .magnet import MAGNeT
from .unet import DiffusionUnet
