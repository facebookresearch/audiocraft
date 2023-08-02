# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from queue import Queue, Empty
import signal
import sys
import threading
import traceback

logger = logging.getLogger(__name__)


class DeadlockDetect:
    def __init__(self, use: bool = False, timeout: float = 120.):
        self.use = use
        self.timeout = timeout
        self._queue: Queue = Queue()

    def update(self, stage: str):
        if self.use:
            self._queue.put(stage)

    def __enter__(self):
        if self.use:
            self._thread = threading.Thread(target=self._detector_thread)
            self._thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use:
            self._queue.put(None)
            self._thread.join()

    def _detector_thread(self):
        logger.debug("Deadlock detector started")
        last_stage = "init"
        while True:
            try:
                stage = self._queue.get(timeout=self.timeout)
            except Empty:
                break
            if stage is None:
                logger.debug("Exiting deadlock detector thread")
                return
            else:
                last_stage = stage
        logger.error("Deadlock detector timed out, last stage was %s", last_stage)
        for th in threading.enumerate():
            print(th, file=sys.stderr)
            traceback.print_stack(sys._current_frames()[th.ident])
            print(file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGKILL)
