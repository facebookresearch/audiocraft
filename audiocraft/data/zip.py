# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utility for reading some info from inside a zip file.
"""

import typing
import zipfile

from dataclasses import dataclass
from functools import lru_cache
from typing_extensions import Literal


DEFAULT_SIZE = 32
MODE = Literal['r', 'w', 'x', 'a']


@dataclass(order=True)
class PathInZip:
    """Hold a path of file within a zip file.

    Args:
        path (str): The convention is <path_to_zip>:<relative_path_inside_zip>.
            Let's assume there is a zip file /some/location/foo.zip
            and inside of it is a json file located at /data/file1.json,
            Then we expect path = "/some/location/foo.zip:/data/file1.json".
    """

    INFO_PATH_SEP = ':'
    zip_path: str
    file_path: str

    def __init__(self, path: str) -> None:
        split_path = path.split(self.INFO_PATH_SEP)
        assert len(split_path) == 2
        self.zip_path, self.file_path = split_path

    @classmethod
    def from_paths(cls, zip_path: str, file_path: str):
        return cls(zip_path + cls.INFO_PATH_SEP + file_path)

    def __str__(self) -> str:
        return self.zip_path + self.INFO_PATH_SEP + self.file_path


def _open_zip(path: str, mode: MODE = 'r'):
    return zipfile.ZipFile(path, mode)


_cached_open_zip = lru_cache(DEFAULT_SIZE)(_open_zip)


def set_zip_cache_size(max_size: int):
    """Sets the maximal LRU caching for zip file opening.

    Args:
        max_size (int): the maximal LRU cache.
    """
    global _cached_open_zip
    _cached_open_zip = lru_cache(max_size)(_open_zip)


def open_file_in_zip(path_in_zip: PathInZip, mode: str = 'r') -> typing.IO:
    """Opens a file stored inside a zip and returns a file-like object.

    Args:
        path_in_zip (PathInZip): A PathInZip object representing the file to return a file-like object of.
        mode (str): The mode in which to open the file with.
    Returns:
        A file-like object for PathInZip.
    """
    zf = _cached_open_zip(path_in_zip.zip_path)
    return zf.open(path_in_zip.file_path)
