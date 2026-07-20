"""Abstract base class for dataset caching mechanisms."""

from __future__ import annotations

import atexit
import logging
import os
import time
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import tqdm
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from nntools.dataset.abstract_image_dataset import AbstractImageDataset

logger = logging.getLogger(__name__)


class SharedMemoryHandle:
    """
    RAII wrapper for shared memory with proper lifecycle management.

    Handles creation, attachment, and cleanup of shared memory segments.
    Only the creator process will unlink the memory on cleanup.
    Fork-safe: tracks the PID that created/attached to prevent issues
    when forked processes inherit handles.
    """

    def __init__(
        self,
        name: str,
        size: int,
        is_creator: bool,
        max_attach_retries: int = 50,
        retry_delay: float = 0.1,
    ):
        """
        Initialize shared memory handle.

        Args:
            name: Unique identifier for the shared memory segment
            size: Size in bytes
            is_creator: Whether this process should create (True) or attach (False)
            max_attach_retries: Maximum attempts to attach to existing memory
            retry_delay: Delay between retry attempts in seconds
        """
        self.name = name
        self.size = size
        self._is_creator = is_creator
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._closed = False
        self._owner_pid = os.getpid()  # Track which process owns this handle

        if is_creator:
            self._create(size)
        else:
            self._attach(size, max_attach_retries, retry_delay)

        atexit.register(self.close)

    def _create(self, size: int) -> None:
        """Create new shared memory, cleaning up stale segments if necessary."""
        try:
            self._shm = shared_memory.SharedMemory(name=self.name, size=size, create=True)
            logger.debug(f"Created shared memory '{self.name}' ({size:,} bytes)")
        except FileExistsError:
            logger.warning(f"Stale shared memory '{self.name}' found, cleaning up")
            self._cleanup_stale()
            self._shm = shared_memory.SharedMemory(name=self.name, size=size, create=True)
            logger.debug(f"Created shared memory '{self.name}' after cleanup ({size:,} bytes)")

    def _cleanup_stale(self) -> None:
        """Remove stale shared memory from a previous crashed process."""
        try:
            stale = shared_memory.SharedMemory(name=self.name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass  # Already cleaned up

    def _attach(self, expected_size: int, max_retries: int, retry_delay: float) -> None:
        """Attach to existing shared memory with retry logic for synchronization."""
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                self._shm = shared_memory.SharedMemory(name=self.name, create=False)

                if self._shm.size != expected_size:
                    actual_size = self._shm.size
                    self._shm.close()
                    self._shm = None
                    raise ValueError(
                        f"Shared memory '{self.name}' size mismatch: expected {expected_size:,}, got {actual_size:,}"
                    )

                logger.debug(f"Attached to shared memory '{self.name}' (attempt {attempt + 1})")
                return

            except FileNotFoundError as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        raise TimeoutError(
            f"Failed to attach to shared memory '{self.name}' after {max_retries} attempts"
        ) from last_error

    @property
    def buffer(self) -> memoryview:
        """Access the shared memory buffer."""
        if self._shm is None or self._closed:
            raise RuntimeError(f"Shared memory '{self.name}' is not available")
        return self._shm.buf

    def as_array(self, dtype: np.dtype, shape: tuple) -> np.ndarray:
        """Create a numpy array view of the shared memory."""
        return np.ndarray(shape, dtype=dtype, buffer=self.buffer)

    def close(self) -> None:
        """Close and optionally unlink the shared memory."""
        if self._closed or self._shm is None:
            return

        # Only the original process should close/unlink
        # Forked processes inherit handles but shouldn't manage them
        if os.getpid() != self._owner_pid:
            logger.debug(
                f"Skipping cleanup of '{self.name}' in forked process (owner={self._owner_pid}, current={os.getpid()})"
            )
            return

        try:
            self._shm.close()
        except Exception as e:
            logger.error(f"Error closing shared memory '{self.name}': {e}")

        if self._is_creator:
            try:
                self._shm.unlink()
                logger.debug(f"Unlinked shared memory '{self.name}'")
            except FileNotFoundError:
                pass  # Already unlinked
            except Exception as e:
                logger.error(f"Error unlinking shared memory '{self.name}': {e}")

        self._closed = True

        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __enter__(self) -> "SharedMemoryHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AbstractCache(ABC):
    """
    Abstract base class for dataset caching.

    Provides common infrastructure for tracking cached items and
    defines the interface that concrete cache implementations must follow.
    """

    def __init__(self, dataset: "AbstractImageDataset"):
        """
        Initialize the cache.

        Args:
            dataset: The dataset instance to cache
        """
        self._dataset = dataset
        self._is_initialized = False
        self._shared_memory_handles: List[SharedMemoryHandle] = []
        self._is_item_cached: Optional[np.ndarray] = None

        # Configuration for auto_cache
        self._auto_cache_num_workers = 8
        self._auto_cache_batch_size = 32

    @property
    def dataset(self) -> "AbstractImageDataset":
        """The dataset being cached."""
        return self._dataset

    @property
    def is_initialized(self) -> bool:
        """Whether the cache has been initialized."""
        return self._is_initialized

    @property
    def dataset_id(self) -> str:
        """Unique identifier for the dataset."""
        if not self._dataset.id:
            raise ValueError("Dataset must have a non-empty 'id' for caching. Set dataset.id before using cache.")
        return self._dataset.id

    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset."""
        return self._dataset.real_length

    @property
    def is_item_cached(self) -> np.ndarray:
        """Boolean array tracking which items have been cached."""
        if self._is_item_cached is None:
            raise RuntimeError("Cache not initialized. Call init_cache() first.")
        return self._is_item_cached

    def _shm_name(self, key: str) -> str:
        """Generate a unique shared memory name for a given key."""
        return f"nntools_{self.dataset_id}_{key}"

    def _create_shared_memory(
        self,
        key: str,
        size: int,
        is_creator: bool,
    ) -> SharedMemoryHandle:
        """
        Create or attach to shared memory and track the handle.

        Args:
            key: Identifier for this memory segment
            size: Size in bytes
            is_creator: Whether this process creates the memory

        Returns:
            SharedMemoryHandle for the created/attached memory
        """
        handle = SharedMemoryHandle(
            name=self._shm_name(key),
            size=size,
            is_creator=is_creator,
        )
        self._shared_memory_handles.append(handle)
        return handle

    def _init_item_tracking_shared(self, is_creator: bool) -> None:
        """Initialize shared item tracking array across processes."""
        size = self.num_samples  # bool = 1 byte per item
        handle = self._create_shared_memory("is_item_cached", size, is_creator)
        self._is_item_cached = handle.as_array(dtype=np.bool_, shape=(self.num_samples,))

        if is_creator:
            self._is_item_cached[:] = False

    def _init_item_tracking_local(self) -> None:
        """Initialize local (non-shared) item tracking array."""
        self._is_item_cached = np.zeros(self.num_samples, dtype=np.bool_)

    def configure_auto_cache(self, num_workers: int = 8, batch_size: int = 32) -> None:
        """
        Configure parameters for auto_cache operation.

        Args:
            num_workers: Number of parallel workers for data loading
            batch_size: Batch size for data loading
        """
        self._auto_cache_num_workers = num_workers
        self._auto_cache_batch_size = batch_size

    def auto_cache(self, show_progress: bool = True) -> None:
        """
        Automatically populate the entire cache.

        Args:
            show_progress: Whether to show a progress bar
        """
        self.init_cache()

        dataloader = DataLoader(
            self._dataset,
            num_workers=self._auto_cache_num_workers,
            batch_size=self._auto_cache_batch_size,
            pin_memory=False,
            shuffle=False,
        )

        iterator = tqdm.tqdm(dataloader, desc="Caching dataset") if show_progress else dataloader

        for _ in iterator:
            pass

    def cleanup(self) -> None:
        """Explicitly release all resources."""
        for handle in self._shared_memory_handles:
            handle.close()
        self._shared_memory_handles.clear()
        self._is_item_cached = None
        self._is_initialized = False

    def __del__(self):
        # Only cleanup if we're in the original process that created/attached
        # This prevents issues with forked processes
        import multiprocessing as mp

        if mp.current_process().name == "MainProcess":
            self.cleanup()

    @staticmethod
    def get_worker_init_fn(cache: "AbstractCache"):
        """
        Get a worker_init_fn for DataLoader that initializes the cache.

        Usage:
            cache = MemoryCache(dataset)
            cache.init_cache()  # Initialize in main process first!

            dataloader = DataLoader(
                dataset,
                num_workers=4,
                worker_init_fn=AbstractCache.get_worker_init_fn(cache)
            )

        Args:
            cache: The cache instance to initialize in workers

        Returns:
            A function suitable for DataLoader's worker_init_fn parameter
        """

        def worker_init_fn(worker_id: int):
            # Re-initialize cache in worker (will attach to existing shm)
            cache._is_initialized = False  # Force re-initialization
            cache.init_cache()

        return worker_init_fn

    @abstractmethod
    def init_cache(self) -> None:
        """Initialize the cache. Must be called before accessing items."""
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> Dict[str, Any]:
        """Retrieve an item from cache, loading it if necessary."""
        pass

    @abstractmethod
    def remap(self, old_key: str, new_key: str) -> None:
        """Rename a key in the cached data."""
        pass
