# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile


class TempDirMixin:
    """Mixin to provide easy access to temp dir.
    """

    temp_dir_ = None

    @classmethod
    def get_base_temp_dir(cls):
        # If AUDIOCRAFT_TEST_DIR is set, use it instead of temporary directory.
        # this is handy for debugging.
        key = "AUDIOCRAFT_TEST_DIR"
        if key in os.environ:
            return os.environ[key]
        if cls.temp_dir_ is None:
            cls.temp_dir_ = tempfile.TemporaryDirectory()
        return cls.temp_dir_.name

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir_ is not None:
            try:
                cls.temp_dir_.cleanup()
                cls.temp_dir_ = None
            except PermissionError:
                # On Windows there is a know issue with `shutil.rmtree`,
                # which fails intermittently.
                # https://github.com/python/cpython/issues/74168
                # Following the above thread, we ignore it.
                pass
        super().tearDownClass()

    @property
    def id(self):
        return self.__class__.__name__

    def get_temp_path(self, *paths):
        temp_dir = os.path.join(self.get_base_temp_dir(), self.id)
        path = os.path.join(temp_dir, *paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_temp_dir(self, *paths):
        temp_dir = os.path.join(self.get_base_temp_dir(), self.id)
        path = os.path.join(temp_dir, *paths)
        os.makedirs(path, exist_ok=True)
        return path
