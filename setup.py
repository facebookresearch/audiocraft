# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Legacy setup.py for backward compatibility.
All configuration is now in pyproject.toml and should be managed with UV.
"""

from setuptools import setup

# This setup.py is kept for backward compatibility
# All project configuration is now in pyproject.toml
# Use `uv build` or `uv sync` instead of `pip install .`
setup()
