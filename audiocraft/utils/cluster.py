# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for SLURM configuration and cluster settings.
"""

from enum import Enum
import os
import socket
import typing as tp

import omegaconf


class ClusterType(Enum):
    AWS = "aws"
    FAIR = "fair"
    RSC = "rsc"
    LOCAL_DARWIN = "darwin"
    DEFAULT = "default"  # used for any other cluster.


def _guess_cluster_type() -> ClusterType:
    uname = os.uname()
    fqdn = socket.getfqdn()
    if uname.sysname == "Linux" and (uname.release.endswith("-aws") or ".ec2" in fqdn):
        return ClusterType.AWS

    if fqdn.endswith(".fair"):
        return ClusterType.FAIR

    if fqdn.endswith(".facebook.com"):
        return ClusterType.RSC

    if uname.sysname == "Darwin":
        return ClusterType.LOCAL_DARWIN

    return ClusterType.DEFAULT


def get_cluster_type(
    cluster_type: tp.Optional[ClusterType] = None,
) -> tp.Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


def get_slurm_parameters(
    cfg: omegaconf.DictConfig, cluster_type: tp.Optional[ClusterType] = None
) -> omegaconf.DictConfig:
    """Update SLURM parameters in configuration based on cluster type.
    If the cluster type is not specify, it infers it automatically.
    """
    from ..environment import AudioCraftEnvironment
    cluster_type = get_cluster_type(cluster_type)
    # apply cluster-specific adjustments
    if cluster_type == ClusterType.AWS:
        cfg["mem_per_gpu"] = None
        cfg["constraint"] = None
        cfg["setup"] = []
    elif cluster_type == ClusterType.RSC:
        cfg["mem_per_gpu"] = None
        cfg["setup"] = []
        cfg["constraint"] = None
        cfg["partition"] = "learn"
    slurm_exclude = AudioCraftEnvironment.get_slurm_exclude()
    if slurm_exclude is not None:
        cfg["exclude"] = slurm_exclude
    return cfg
