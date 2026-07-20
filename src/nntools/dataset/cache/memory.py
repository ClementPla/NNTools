"""In-memory cache implementations for dataset caching."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

import numpy as np

from nntools.dataset.cache.abstract_cache import AbstractCache

if TYPE_CHECKING:
    from nntools.dataset.abstract_image_dataset import AbstractImageDataset

logger = logging.getLogger(__name__)


# Environment variable set by main process to signal shared memory is ready
_NNTOOLS_SHM_READY_ENV = "NNTOOLS_SHM_READY_{dataset_id}"


def _is_dataloader_worker() -> bool:
    """
    Check if we're running inside a DataLoader worker process.

    Note: get_worker_info() only returns non-None when called during
    the actual data loading (__getitem__), not during worker initialization.
    We use an environment variable marker instead.
    """
    # Check if we're in a spawned/forked worker by looking at process name
    proc_name = mp.current_process().name
    if proc_name.startswith("SpawnPoolWorker") or proc_name.startswith("ForkPoolWorker"):
        return True
    if "DataLoader" in proc_name and "Worker" in proc_name:
        return True
    # Also check for the torch worker pattern
    if proc_name.startswith("Process-"):
        return True
    return False


def _get_distributed_rank() -> int:
    """Get rank in distributed training (0 if not distributed)."""
    for env_var in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        rank = os.environ.get(env_var)
        if rank is not None:
            return int(rank)
    return 0


def _check_shm_ready(dataset_id: str) -> bool:
    """Check if main process has signaled that shared memory is ready."""
    env_var = _NNTOOLS_SHM_READY_ENV.format(dataset_id=dataset_id)
    return os.environ.get(env_var) == "1"


def _signal_shm_ready(dataset_id: str) -> None:
    """Signal that shared memory has been created and is ready."""
    env_var = _NNTOOLS_SHM_READY_ENV.format(dataset_id=dataset_id)
    os.environ[env_var] = "1"


class MemoryCache(AbstractCache):
    """
    Cache that stores dataset items in shared memory.

    Enables efficient data sharing between multiple DataLoader worker
    processes without duplicating memory.

    IMPORTANT USAGE PATTERN:

    For multi-worker DataLoader, you MUST initialize the cache in the
    main process BEFORE creating/iterating the DataLoader:

        dataset = MyDataset(..., use_cache=True, id="my_dataset")
        dataset.init_cache()  # <-- REQUIRED: creates shared memory

        dataloader = DataLoader(dataset, num_workers=4)
        for batch in dataloader:  # Workers will attach to existing shm
            ...

    If you forget to call init_cache(), workers will fail with:
    "Failed to attach to shared memory ... after N attempts"

    Attributes:
        use_shared_memory: Whether to use shared memory (True) or
            process-local memory (False)
    """

    def __init__(self, dataset: "AbstractImageDataset", use_shared_memory: bool = True):
        """
        Initialize the memory cache.

        Args:
            dataset: The dataset to cache
            use_shared_memory: If True, use shared memory for multi-process
                access. If False, each process maintains its own copy.
        """
        super().__init__(dataset)
        self.use_shared_memory = use_shared_memory
        self._cache_arrays: Optional[Dict[str, np.ndarray]] = None
        self._is_creator = False
        self._array_metadata: Dict[str, Dict[str, Any]] = {}
        self._in_memory_keys: Set[str] = set()
        self._fallback_to_local = False

    def init_cache(self) -> None:
        """
        Initialize the cache by creating or attaching to shared memory.

        For shared memory mode with DataLoader workers:
        - Main process creates shared memory and signals readiness
        - DataLoader workers detect the signal and attach
        - If signal not found, workers fall back to local memory with warning

        This method is idempotent and safe to call multiple times.
        """
        if self._is_initialized:
            return

        self._validate_dataset_configuration()

        # Determine array shapes and dtypes from first sample
        sample_data = self._load_sample_data(0)

        # Determine our role
        is_worker = _is_dataloader_worker()
        dist_rank = _get_distributed_rank()
        shm_ready = _check_shm_ready(self.dataset_id)

        if is_worker:
            if shm_ready:
                # Main process created shared memory - attach to it
                self._is_creator = False
                logger.debug(f"Worker {mp.current_process().name} attaching to shared memory")
            else:
                # Main process didn't initialize cache - fall back to local memory
                logger.warning(
                    f"DataLoader worker cannot find shared memory for '{self.dataset_id}'. "
                    f"Did you forget to call dataset.init_cache() before creating the DataLoader? "
                    f"Falling back to local memory (reduced performance)."
                )
                self._fallback_to_local = True
                self.use_shared_memory = False
        elif dist_rank == 0:
            # Main process or distributed rank 0 - create shared memory
            self._is_creator = True
        else:
            # Non-zero distributed rank - attach
            self._is_creator = False

        # Initialize item tracking
        if self.use_shared_memory:
            self._init_item_tracking_shared(self._is_creator)
        else:
            self._init_item_tracking_local()

        # Create cache arrays
        self._cache_arrays = {}
        self._array_metadata = {}
        self._in_memory_keys = set()

        for key, value in sample_data.items():
            self._create_cache_array(key, value)

        self._is_initialized = True

        # Signal to workers that shared memory is ready
        if self._is_creator and self.use_shared_memory:
            _signal_shm_ready(self.dataset_id)

        role = "creator" if self._is_creator else ("fallback" if self._fallback_to_local else "worker")
        mem_type = "local" if not self.use_shared_memory else "shared"
        logger.info(
            f"Initialized {mem_type} memory cache for {self.num_samples} samples "
            f"(role={role}, process={mp.current_process().name})"
        )

    def _validate_dataset_configuration(self) -> None:
        """Validate that the dataset is properly configured for caching."""
        if not self._dataset.auto_resize and not self._dataset.auto_pad:
            logger.warning(
                "Cache used with auto_resize=False and auto_pad=False. "
                "Ensure all images have identical dimensions to avoid errors."
            )

    def _load_sample_data(self, index: int) -> Dict[str, Any]:
        """Load and preprocess a sample to determine data structure."""
        data = self._dataset.read_from_disk(index)
        return self._dataset.precompose_data(data)

    def _create_cache_array(self, key: str, sample_value: Any) -> None:
        """
        Create a cache array for a given key based on sample value.

        Args:
            key: The data key (e.g., 'image', 'mask')
            sample_value: A sample value to determine dtype and shape
        """
        if not isinstance(sample_value, np.ndarray):
            # Non-array values cannot live in cross-process shared memory
            # (object arrays are process-local). Read them from dataset.gts
            # on access instead, which is inherited identically by every
            # process. This requires the key to be present in gts.
            if key not in self._dataset.gts:
                raise ValueError(
                    f"Key '{key}' is not a numpy array and not found in "
                    f"dataset.gts. Non-array values must be stored in gts to be "
                    f"cacheable across processes."
                )
            self._array_metadata[key] = {
                "is_array": False,
                "dtype": type(sample_value),
            }
            self._in_memory_keys.add(key)
            return

        # Array value - create appropriately sized cache
        item_shape = sample_value.shape
        item_dtype = sample_value.dtype
        full_shape = (self.num_samples, *item_shape)
        total_bytes = int(np.prod(full_shape)) * item_dtype.itemsize

        self._array_metadata[key] = {
            "is_array": True,
            "dtype": item_dtype,
            "item_shape": item_shape,
            "full_shape": full_shape,
        }

        if self.use_shared_memory:
            handle = self._create_shared_memory(key, total_bytes, self._is_creator)
            array = handle.as_array(item_dtype, full_shape)

            # NOTE: no eager `array[:] = 0` here. POSIX shared memory is already
            # zero-filled by the OS, and touching every page up front would force
            # the full (potentially multi-GB) allocation before serving a single
            # sample, defeating the lazy-caching purpose. Pages are populated
            # lazily as items are cached.
            self._cache_arrays[key] = array
        else:
            self._cache_arrays[key] = np.zeros(full_shape, dtype=item_dtype)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Retrieve an item from cache.

        If the item is not cached, loads it from disk and caches it.

        Args:
            item: Index of the item to retrieve

        Returns:
            Dictionary mapping keys to their values for this item
        """
        if not self._is_initialized:
            raise RuntimeError("Cache not initialized. Call init_cache() first.")

        if self.is_item_cached[item]:
            return self._get_cached_item(item)

        return self._cache_and_return(item)

    def _get_cached_item(self, item: int) -> Dict[str, Any]:
        """Retrieve an already-cached item."""
        result = {}
        for key, array in self._cache_arrays.items():
            value = array[item]
            # Return a copy to prevent accidental modification of cache
            if isinstance(value, np.ndarray):
                result[key] = value.copy()
            else:
                result[key] = value
        # Non-array keys are read from dataset.gts (shared identically across
        # processes) rather than a process-local cache array.
        for key in self._in_memory_keys:
            result[key] = self._dataset.gts[key][item]
        return result

    def _cache_and_return(self, item: int) -> Dict[str, Any]:
        """Load item from disk, cache it, and return."""
        data = self._load_sample_data(item)

        for key, value in data.items():
            if key in self._cache_arrays:
                self._cache_arrays[key][item] = value

        # Mark as cached after all data is written
        # Note: This is not atomic, but duplicate caching is harmless
        self.is_item_cached[item] = True

        return data

    def remap(self, old_key: str, new_key: str) -> None:
        """
        Rename a key in the cache.

        Args:
            old_key: Current key name
            new_key: New key name
        """
        if self._cache_arrays is None:
            return

        if old_key in self._cache_arrays:
            self._cache_arrays[new_key] = self._cache_arrays.pop(old_key)

        if old_key in self._array_metadata:
            self._array_metadata[new_key] = self._array_metadata.pop(old_key)

        if old_key in self._in_memory_keys:
            self._in_memory_keys.discard(old_key)
            self._in_memory_keys.add(new_key)

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics for the cache.

        Returns:
            Dictionary with memory usage in bytes per key
        """
        if self._cache_arrays is None:
            return {}

        usage = {}
        for key, array in self._cache_arrays.items():
            if isinstance(array, np.ndarray):
                usage[key] = array.nbytes
            else:
                usage[key] = 0
        return usage

    def cleanup(self) -> None:
        """Release all cache resources."""
        self._cache_arrays = None
        self._array_metadata = {}
        self._in_memory_keys = set()
        self._is_creator = False
        super().cleanup()


class LocalMemoryCache(MemoryCache):
    """
    Memory cache that uses process-local memory instead of shared memory.

    Each DataLoader worker will have its own copy of cached data.
    Useful when shared memory is unavailable or when dataset is small.
    """

    def __init__(self, dataset: "AbstractImageDataset"):
        """
        Initialize the local memory cache.

        Args:
            dataset: The dataset to cache
        """
        super().__init__(dataset, use_shared_memory=False)
