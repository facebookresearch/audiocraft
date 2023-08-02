# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Provides cluster and tools configuration across clusters (slurm, dora, utilities).
"""

import logging
import os
from pathlib import Path
import re
import typing as tp

import omegaconf

from .utils.cluster import _guess_cluster_type


logger = logging.getLogger(__name__)


class AudioCraftEnvironment:
    """Environment configuration for teams and clusters.

    AudioCraftEnvironment picks compute cluster settings (slurm, dora) from the current running environment
    or declared variable and the loaded team configuration. Additionally, the AudioCraftEnvironment
    provides pointers to a reference folder resolved automatically across clusters that is shared across team members,
    allowing to share sigs or other files to run jobs. Finally, it provides dataset mappers to automatically
    map dataset file paths to new locations across clusters, allowing to use the same manifest of files across cluters.

    The cluster type is identified automatically and base configuration file is read from config/teams.yaml.
    Use the following environment variables to specify the cluster, team or configuration:

        AUDIOCRAFT_CLUSTER (optional): Cluster type to enforce. Useful if the cluster type
            cannot be inferred automatically.
        AUDIOCRAFT_CONFIG (optional): Path to yaml config holding the teams configuration.
            If not set, configuration is read from config/teams.yaml.
        AUDIOCRAFT_TEAM (optional): Name of the team. Recommended to set to your own team.
            Cluster configuration are shared across teams to match compute allocation,
            specify your cluster configuration in the configuration file under a key mapping
            your team name.
    """
    _instance = None
    DEFAULT_TEAM = "default"

    def __init__(self) -> None:
        """Loads configuration."""
        self.team: str = os.getenv("AUDIOCRAFT_TEAM", self.DEFAULT_TEAM)
        cluster_type = _guess_cluster_type()
        cluster = os.getenv(
            "AUDIOCRAFT_CLUSTER", cluster_type.value
        )
        logger.info("Detecting cluster type %s", cluster_type)

        self.cluster: str = cluster

        config_path = os.getenv(
            "AUDIOCRAFT_CONFIG",
            Path(__file__)
            .parent.parent.joinpath("config/teams", self.team)
            .with_suffix(".yaml"),
        )
        self.config = omegaconf.OmegaConf.load(config_path)
        self._dataset_mappers = []
        cluster_config = self._get_cluster_config()
        if "dataset_mappers" in cluster_config:
            for pattern, repl in cluster_config["dataset_mappers"].items():
                regex = re.compile(pattern)
                self._dataset_mappers.append((regex, repl))

    def _get_cluster_config(self) -> omegaconf.DictConfig:
        assert isinstance(self.config, omegaconf.DictConfig)
        return self.config[self.cluster]

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Clears the environment and forces a reload on next invocation."""
        cls._instance = None

    @classmethod
    def get_team(cls) -> str:
        """Gets the selected team as dictated by the AUDIOCRAFT_TEAM env var.
        If not defined, defaults to "labs".
        """
        return cls.instance().team

    @classmethod
    def get_cluster(cls) -> str:
        """Gets the detected cluster.
        This value can be overridden by the AUDIOCRAFT_CLUSTER env var.
        """
        return cls.instance().cluster

    @classmethod
    def get_dora_dir(cls) -> Path:
        """Gets the path to the dora directory for the current team and cluster.
        Value is overridden by the AUDIOCRAFT_DORA_DIR env var.
        """
        cluster_config = cls.instance()._get_cluster_config()
        dora_dir = os.getenv("AUDIOCRAFT_DORA_DIR", cluster_config["dora_dir"])
        logger.warning(f"Dora directory: {dora_dir}")
        return Path(dora_dir)

    @classmethod
    def get_reference_dir(cls) -> Path:
        """Gets the path to the reference directory for the current team and cluster.
        Value is overridden by the AUDIOCRAFT_REFERENCE_DIR env var.
        """
        cluster_config = cls.instance()._get_cluster_config()
        return Path(os.getenv("AUDIOCRAFT_REFERENCE_DIR", cluster_config["reference_dir"]))

    @classmethod
    def get_slurm_exclude(cls) -> tp.Optional[str]:
        """Get the list of nodes to exclude for that cluster."""
        cluster_config = cls.instance()._get_cluster_config()
        return cluster_config.get("slurm_exclude")

    @classmethod
    def get_slurm_partitions(cls, partition_types: tp.Optional[tp.List[str]] = None) -> str:
        """Gets the requested partitions for the current team and cluster as a comma-separated string.

        Args:
            partition_types (list[str], optional): partition types to retrieve. Values must be
                from ['global', 'team']. If not provided, the global partition is returned.
        """
        if not partition_types:
            partition_types = ["global"]

        cluster_config = cls.instance()._get_cluster_config()
        partitions = [
            cluster_config["partitions"][partition_type]
            for partition_type in partition_types
        ]
        return ",".join(partitions)

    @classmethod
    def resolve_reference_path(cls, path: tp.Union[str, Path]) -> Path:
        """Converts reference placeholder in path with configured reference dir to resolve paths.

        Args:
            path (str or Path): Path to resolve.
        Returns:
            Path: Resolved path.
        """
        path = str(path)

        if path.startswith("//reference"):
            reference_dir = cls.get_reference_dir()
            logger.warn(f"Reference directory: {reference_dir}")
            assert (
                reference_dir.exists() and reference_dir.is_dir()
            ), f"Reference directory does not exist: {reference_dir}."
            path = re.sub("^//reference", str(reference_dir), path)

        return Path(path)

    @classmethod
    def apply_dataset_mappers(cls, path: str) -> str:
        """Applies dataset mapping regex rules as defined in the configuration.
        If no rules are defined, the path is returned as-is.
        """
        instance = cls.instance()

        for pattern, repl in instance._dataset_mappers:
            path = pattern.sub(repl, path)

        return path
