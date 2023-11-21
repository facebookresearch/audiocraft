# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from functools import partial
from hashlib import sha1
import logging
from pathlib import Path
import sys
import typing as tp
import zipfile

import flashy
import torch


logger = logging.getLogger(__name__)


def get_full_embed(full_embed: torch.Tensor, x: tp.Any, idx: int, device: tp.Union[str, torch.device]) -> torch.Tensor:
    """Utility function for the EmbeddingCache, returning the full embedding without any chunking.
    This method can be used in case there is no need in extracting a chunk of the full embedding
    read from the cache.

    Args:
        full_embed (torch.Tensor): The full embedding.
        x (any): Batch object from which the full embedding is derived.
        idx (torch.Tensor): Index of object to consider in the batch object.
    Returns:
        full_embed (torch.Tensor): The full embedding
    """
    return full_embed.to(device)


class EmbeddingCache:
    """Cache around embeddings computation for faster execution.
    The EmbeddingCache is storing pre-computed embeddings on disk and provides a simple API
    to retrieve the pre-computed embeddings on full inputs and extract only a given chunk
    using a user-provided function. When the cache is warm (all embeddings are pre-computed),
    the EmbeddingCache allows for faster training as it removes the need of computing the embeddings.
    Additionally, it provides in-memory cache around the loaded embeddings to limit IO footprint
    and synchronization points in the forward calls.

    Args:
        cache_path (Path): Path to folder where all pre-computed embeddings are saved on disk.
        device (str or torch.device): Device on which the embedding is returned.
        compute_embed_fn (callable[[Path, any, int], torch.Tensor], optional): Function to compute
            the embedding from a given object and path. This user provided function can compute the
            embedding from the provided object or using the provided path as entry point. The last parameter
            specify the index corresponding to the current embedding in the object that can represent batch metadata.
        extract_embed_fn (callable[[torch.Tensor, any, int], torch.Tensor], optional): Function to extract
            the desired embedding chunk from the full embedding loaded from the cache. The last parameter
            specify the index corresponding to the current embedding in the object that can represent batch metadata.
            If not specified, will return the full embedding unmodified.
    """
    def __init__(self, cache_path: tp.Union[str, Path], device: tp.Union[str, torch.device],
                 compute_embed_fn: tp.Callable[[Path, tp.Any, int], torch.Tensor],
                 extract_embed_fn: tp.Optional[tp.Callable[[torch.Tensor, tp.Any, int], torch.Tensor]] = None):
        self.cache_path = Path(cache_path)
        self.device = device
        self._compute_embed_fn = compute_embed_fn
        self._extract_embed_fn: tp.Callable[[torch.Tensor, tp.Any, int], torch.Tensor]
        if extract_embed_fn is not None:
            self._extract_embed_fn = extract_embed_fn
        else:
            self._extract_embed_fn = partial(get_full_embed, device=device)
        if self.cache_path is not None:
            self.cache_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"Cache instantiated at: {self.cache_path}")
            self.pool = ThreadPoolExecutor(8)
            self.pool.__enter__()
        self._current_batch_cache: dict = {}
        self._memory_cache: dict = {}

    def _get_cache_path(self, path: tp.Union[Path, str]):
        """Get cache path for the given file path."""
        sig = sha1(str(path).encode()).hexdigest()
        return self.cache_path / sig

    @staticmethod
    def _get_full_embed_from_cache(cache: Path):
        """Loads full pre-computed embedding from the cache."""
        try:
            embed = torch.load(cache, 'cpu')
        except Exception as exc:
            logger.error("Error loading %s: %r", cache, exc)
            embed = None
        return embed

    def get_embed_from_cache(self, paths: tp.List[Path], x: tp.Any) -> torch.Tensor:
        """Get embedding from cache, computing and storing it to cache if not already cached.
        The EmbeddingCache first tries to load the embedding from the in-memory cache
        containing the pre-computed chunks populated through `populate_embed_cache`.
        If not found, the full embedding is computed and stored on disk to be later accessed
        to populate the in-memory cache, and the desired embedding chunk is extracted and returned.

        Args:
            paths (list[Path or str]): List of paths from where the embeddings can be loaded.
            x (any): Object from which the embedding is extracted.
        """
        embeds = []
        for idx, path in enumerate(paths):
            cache = self._get_cache_path(path)
            if cache in self._current_batch_cache:
                embed = self._current_batch_cache[cache]
            else:
                full_embed = self._compute_embed_fn(path, x, idx)
                try:
                    with flashy.utils.write_and_rename(cache, pid=True) as f:
                        torch.save(full_embed.cpu(), f)
                except Exception as exc:
                    logger.error('Error saving embed %s (%s): %r', cache, full_embed.shape, exc)
                else:
                    logger.info('New embed cache saved: %s (%s)', cache, full_embed.shape)
                    embed = self._extract_embed_fn(full_embed, x, idx)
            embeds.append(embed)
        embed = torch.stack(embeds, dim=0)
        return embed

    def populate_embed_cache(self, paths: tp.List[Path], x: tp.Any) -> None:
        """Populate in-memory caches for embeddings reading from the embeddings stored on disk.
        The in-memory caches consist in a cache for the full embedding and another cache for the
        final embedding chunk. Such caches are used to limit the IO access when computing the actual embeddings
        and reduce the IO footprint and synchronization points during forward passes.

        Args:
            paths (list[Path]): List of paths from where the embeddings can be loaded.
            x (any): Object from which the embedding is extracted.
        """
        self._current_batch_cache.clear()
        if self.cache_path is not None:
            futures: list = []
            for path in paths:
                assert path is not None, "Path is required for computation from cache"
                cache = self._get_cache_path(path)
                if cache in self._memory_cache or not cache.exists():
                    futures.append(None)
                else:
                    futures.append(self.pool.submit(EmbeddingCache._get_full_embed_from_cache, cache))
            for idx, (path, future) in enumerate(zip(paths, futures)):
                assert path is not None
                cache = self._get_cache_path(path)
                full_embed = None
                if future is None:
                    if cache in self._memory_cache:
                        full_embed = self._memory_cache[cache]
                else:
                    full_embed = future.result()
                    if full_embed is not None:
                        self._memory_cache[cache] = full_embed
                        full_embed = full_embed.to(self.device)
                if full_embed is not None:
                    embed = self._extract_embed_fn(full_embed, x, idx)
                    self._current_batch_cache[cache] = embed


class CachedBatchWriter:
    """Write pre computed caches for mini batches. This can
    make loading a lot more efficient depending on your filesystem.

    Args:
        cache_folder (Path): folder in which the cached minibatches
            will be stored.

    Inside cache folder, the structure is the following:
    `epoch_number / update_number.zip`
    And the zip file contains one entry per batch item.

    It is possible to use the cache with a batch size smaller than
    created with but obviously not larger. Make sure to call the
    `start_epoch(epoch)` method for indicating changes of epochs.

    See the grid `audiocraft/grids/musicgen/musicgen_warmup_cache.py`
    for an example of how to warmup the cache.
    """
    def __init__(self, cache_folder: Path):
        self.cache_folder = cache_folder
        self._current_epoch: tp.Optional[int] = None
        self._current_index = 0

    def start_epoch(self, epoch: int):
        """Call at the beginning of each epoch.
        """
        self._current_epoch = epoch
        self._current_index = 0
        self._zip_path.parent.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _get_zip_path(cache_folder: Path, epoch: int, index: int):
        return cache_folder / f"{epoch:05d}" / f"{index:06d}.zip"

    @property
    def _zip_path(self):
        assert self._current_epoch is not None
        return CachedBatchWriter._get_zip_path(self.cache_folder, self._current_epoch, self._current_index)

    def save(self, *content):
        """Save one mini batch. This function is distributed-aware
        and will automatically merge all the items from the different
        workers.
        """
        all_contents = []
        for rank in range(flashy.distrib.world_size()):
            their_content = flashy.distrib.broadcast_object(content, src=rank)
            all_contents.append(their_content)

        if flashy.distrib.is_rank_zero():
            idx = 0
            with flashy.utils.write_and_rename(self._zip_path) as tmp:
                with zipfile.ZipFile(tmp, 'w') as zf:
                    for content in all_contents:
                        for vals in zip(*content):
                            with zf.open(f'{idx}', 'w') as f:  # type: ignore
                                torch.save(vals, f)
                            idx += 1
        flashy.distrib.barrier()
        self._current_index += 1


class CachedBatchLoader:
    """Loader for cached mini-batches dumped with `CachedBatchWriter`.

    Args:
        cache_folder (Path): folder in which the cached minibatches are stored.
        batch_size (int): batch size (per GPU) expected.
        num_workers (int): number of workers to use for loading.
        min_length (int): minimum expected length for each epoch. If some
            mini-batches are missing, and error is raised.

    This is iterable just like a regular DataLoader.
    """

    def __init__(self, cache_folder: Path, batch_size: int,
                 num_workers: int = 10, min_length: int = 1):
        self.cache_folder = cache_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_length = min_length
        self._current_epoch: tp.Optional[int] = None
        self.sampler = None  # for compatibility with the regular DataLoader

    def __len__(self):
        path = CachedBatchWriter._get_zip_path(self.cache_folder, self._current_epoch or 0, 0).parent
        return len([p for p in path.iterdir() if p.suffix == ".zip"])

    def start_epoch(self, epoch: int):
        """Call at the beginning of each epoch.
        """
        self._current_epoch = epoch

    def _zip_path(self, index: int):
        assert self._current_epoch is not None
        return CachedBatchWriter._get_zip_path(self.cache_folder, self._current_epoch, index)

    def _load_one(self, index: int):
        zip_path = self._zip_path(index)
        if not zip_path.exists():
            if index < self.min_length:
                raise RuntimeError(f"Cache should have at least {self.min_length} batches, but {index} doesn't exist")

            return None
        mode = "rb" if sys.version_info >= (3, 9) else "r"
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                rank = flashy.distrib.rank()
                world_size = flashy.distrib.world_size()
                root = zipfile.Path(zf)
                items = list(root.iterdir())
                total_batch_size = self.batch_size * world_size
                if len(items) < total_batch_size:
                    raise RuntimeError(
                        f"The cache can handle a max batch size of {len(items)}, "
                        f"but {total_batch_size} is needed.")
                start = rank * self.batch_size
                items = items[start: start + self.batch_size]
                assert len(items) == self.batch_size
                entries = []
                entries = [torch.load(item.open(mode), 'cpu') for item in items]  # type: ignore
                transposed = zip(*entries)
                out = []
                for part in transposed:
                    assert len(part) > 0
                    if isinstance(part[0], torch.Tensor):
                        out.append(torch.stack(part))
                    else:
                        assert isinstance(part, torch.Tensor)
                        out.append(part)
                return out
        except Exception:
            logger.error("Error when reading zip path %s", zip_path)
            raise

    def __iter__(self):
        """This will yields tuples, exactly as provided to the
        `CachedBatchWriter.save` method.
        """
        pool = ThreadPoolExecutor(self.num_workers)
        next_index = 0
        queue = deque()

        def _get_next():
            nonlocal next_index
            r = queue.popleft().result()
            if r is None:
                return None
            else:
                queue.append(pool.submit(self._load_one, next_index))
                next_index += 1
            return r

        with pool:
            # fill the buffer of fetching jobs.
            for _ in range(2 * self.num_workers):
                queue.append(pool.submit(self._load_one, next_index))
                next_index += 1
            while True:
                batch = _get_next()
                if batch is None:
                    return
                yield batch
