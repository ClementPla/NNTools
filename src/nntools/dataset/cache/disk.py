"""Disk-based cache implementation for dataset caching."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

import cv2
import numpy as np

from nntools.dataset.cache.abstract_cache import AbstractCache
from nntools.utils.io import read_image, save_image
from nntools.utils.misc import (
    can_be_stored_as_image,
    convert_to_image,
    is_image,
    revert_image_to_original_dtype,
)

if TYPE_CHECKING:
    from nntools.dataset.abstract_image_dataset import AbstractImageDataset

logger = logging.getLogger(__name__)


def _is_rank_zero() -> bool:
    """Check if current process is rank zero (main process)."""
    for env_var in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        rank = os.environ.get(env_var)
        if rank is not None:
            return int(rank) == 0
    return True


@dataclass
class CacheKeyMetadata:
    """Metadata for a cached data key."""

    cache_folder: Path
    is_native_image: bool = False
    can_store_as_image: bool = False
    original_dtype: Optional[np.dtype] = None

    @property
    def storage_format(self) -> str:
        """Determine the storage format for this key."""
        if self.is_native_image:
            return "image"
        elif self.can_store_as_image:
            return "image_converted"
        else:
            return "numpy"


class DiskCache(AbstractCache):
    """
    Cache that stores preprocessed dataset items on disk.

    Useful for expensive preprocessing operations that should only
    be performed once. Cached data persists across program restarts.

    Storage formats:
    - Native images: Saved as-is with original format
    - Convertible arrays: Saved as PNG with dtype metadata
    - Other arrays: Saved as .npy files
    """

    def __init__(self, dataset: "AbstractImageDataset"):
        """
        Initialize the disk cache.

        Args:
            dataset: The dataset to cache
        """
        super().__init__(dataset)

        self._root_cache_folder: Optional[Path] = None
        self._key_metadata: Dict[str, CacheKeyMetadata] = {}
        self._in_memory_keys: Set[str] = set()

        # Lock for thread-safe cache writes
        self._write_lock = threading.Lock()

        # Track folder creation to avoid redundant filesystem checks
        self._folders_created = False

    def init_cache(self) -> None:
        """
        Initialize the disk cache.

        Creates cache folders and determines storage format for each data key.
        """
        if self._is_initialized:
            return

        self._root_cache_folder = self._get_cache_folder()

        # Load first sample to determine data structure
        sample_data = self._dataset.get_precache_data(0)

        # Initialize item tracking (non-shared for disk cache)
        self._init_item_tracking_local()

        # Analyze each key and set up metadata
        self._setup_key_metadata(sample_data)

        # Create cache folders (rank 0 only in distributed setting)
        self._ensure_cache_folders_exist()

        # Check which items are already cached
        self._scan_existing_cache()

        self._is_initialized = True

        cached_count = self.is_item_cached.sum()
        logger.info(
            f"Initialized disk cache at {self._root_cache_folder} "
            f"({cached_count}/{self.num_samples} items already cached)"
        )

    def _setup_key_metadata(self, sample_data: Dict[str, Any]) -> None:
        """Analyze sample data and create metadata for each key."""
        for key, value in sample_data.items():
            if not isinstance(value, np.ndarray):
                # Non-array values must come from dataset.gts
                if key not in self._dataset.gts:
                    raise ValueError(
                        f"Key '{key}' is not a numpy array and not found in "
                        f"dataset.gts. Non-array values must be stored in gts."
                    )
                self._in_memory_keys.add(key)
            else:
                # Array value - determine storage format
                cache_folder = self._root_cache_folder / key
                self._key_metadata[key] = CacheKeyMetadata(
                    cache_folder=cache_folder,
                    is_native_image=is_image(value),
                    can_store_as_image=can_be_stored_as_image(value),
                    original_dtype=value.dtype if can_be_stored_as_image(value) else None,
                )

    def _ensure_cache_folders_exist(self) -> None:
        """Create cache folders if they don't exist."""
        if self._folders_created:
            return

        if not _is_rank_zero():
            # Non-zero ranks wait for rank 0 to create folders
            self._wait_for_folders()
            self._folders_created = True
            return

        for key, metadata in self._key_metadata.items():
            metadata.cache_folder.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created cache folder: {metadata.cache_folder}")

        self._folders_created = True

    def _wait_for_folders(self, timeout: float = 30.0, poll_interval: float = 0.1) -> None:
        """Wait for cache folders to be created by rank 0."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_exist = all(metadata.cache_folder.exists() for metadata in self._key_metadata.values())
            if all_exist:
                return
            time.sleep(poll_interval)

        missing = [str(m.cache_folder) for m in self._key_metadata.values() if not m.cache_folder.exists()]
        raise TimeoutError(f"Timeout waiting for cache folders to be created: {missing}")

    def _scan_existing_cache(self) -> None:
        """Scan cache folders to determine which items are already cached.

        Lists each key folder once and checks membership in-memory instead of
        issuing ``num_samples * num_keys`` individual ``stat`` calls (which run
        once per worker process on init).
        """
        if not self._key_metadata:
            self._is_item_cached[:] = True
            return

        existing_by_key = {
            key: set(os.listdir(metadata.cache_folder)) if metadata.cache_folder.exists() else set()
            for key, metadata in self._key_metadata.items()
        }

        for i in range(self.num_samples):
            self._is_item_cached[i] = all(
                self._get_cache_filepath(i, key, metadata).name in existing_by_key[key]
                for key, metadata in self._key_metadata.items()
            )

    def _check_item_cached(self, item: int) -> bool:
        """Check if all data for an item is cached on disk."""
        for key, metadata in self._key_metadata.items():
            filepath = self._get_cache_filepath(item, key, metadata)
            if not filepath.exists():
                return False
        return True

    def _source_path(self, item: int, key: str) -> str:
        """Return a stable, per-item source path used to name the cache file.

        Keys backed by their own file list (the image, or extra image folders)
        use that path. Everything else (segmentation masks live in gts, derived
        array keys have no file) falls back to the image path, which is 1:1 with
        the item and always present, guaranteeing a collision-free unique name.
        """
        if key in self._dataset.img_filepath:
            paths = self._dataset.img_filepath[key]
        else:
            paths = self._dataset.img_filepath["image"]
        return str(paths[item])

    def _get_cache_filepath(self, item: int, key: str, metadata: CacheKeyMetadata) -> Path:
        """Get the cache file path for a specific item and key."""
        source = self._source_path(item, key)
        base_name = os.path.basename(source)
        stem = Path(base_name).stem

        # Disambiguate files that share a stem (e.g. recursive loading from
        # several subfolders) by appending a short hash of the full source
        # path. Without this, distinct sources collide onto the same cache
        # file and silently overwrite each other.
        unique = hashlib.sha1(source.encode()).hexdigest()[:8]
        base_path = metadata.cache_folder / f"{stem}_{unique}"

        # Native images are always uint8 (see is_image), so PNG stores them
        # losslessly. Using the *source* suffix here would re-encode e.g. a
        # segmentation mask as JPEG when the source image is .jpg, silently
        # corrupting the label map. Always use a lossless container.
        if metadata.is_native_image or metadata.can_store_as_image:
            return base_path.with_suffix(".png")
        else:
            return base_path.with_suffix(".npy")

    def _get_cache_folder(self) -> Path:
        """Determine the cache folder location."""
        # Get root image path to create cache nearby
        first_image_path = Path(self._dataset.img_filepath["image"][0])
        image_folder = first_image_path.parent
        folder_name = image_folder.name

        # Build cache path: parent/.{folder}_cache/{cache_dir}/{id}/{composer_id}
        cache_path = (
            image_folder.parent
            / f".{folder_name}_cache"
            / self._dataset.cache_dir
            / self.dataset_id
            / self._dataset.composer.id
        )

        return cache_path

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Retrieve an item from cache.

        If not cached, loads from disk, caches it, and returns.

        Args:
            item: Index of the item to retrieve

        Returns:
            Dictionary mapping keys to their values
        """
        if not self._is_initialized:
            raise RuntimeError("Cache not initialized. Call init_cache() first.")

        if self._is_item_cached[item]:
            return self._load_from_cache(item)

        return self._cache_and_return(item)

    def _load_from_cache(self, item: int) -> Dict[str, Any]:
        """Load all data for an item from the cache."""
        data = {}

        # Load in-memory values from dataset.gts
        for key in self._in_memory_keys:
            data[key] = self._dataset.gts[key][item]

        # Load cached arrays from disk
        for key, metadata in self._key_metadata.items():
            filepath = self._get_cache_filepath(item, key, metadata)
            data[key] = self._read_cached_file(filepath, metadata)

        return data

    def _read_cached_file(self, filepath: Path, metadata: CacheKeyMetadata) -> np.ndarray:
        """Read a single cached file."""
        if metadata.is_native_image:
            # read_image returns a BGR->RGB view with a negative stride;
            # make it contiguous so downstream (e.g. torch.from_numpy) is happy.
            return np.ascontiguousarray(read_image(filepath, cv2.IMREAD_UNCHANGED))
        elif metadata.can_store_as_image:
            img = read_image(filepath, cv2.IMREAD_UNCHANGED)
            img = np.ascontiguousarray(img)
            return revert_image_to_original_dtype(img, metadata.original_dtype)
        else:
            return np.load(filepath)

    def _cache_and_return(self, item: int) -> Dict[str, Any]:
        """Load item from source, cache to disk, and return."""
        # Double-check in case another process cached it
        if self._check_item_cached(item):
            self._is_item_cached[item] = True
            return self._load_from_cache(item)

        # Load from source
        data = self._dataset.read_from_disk(item)
        data = self._dataset.precompose_data(data)

        # Cache each array to disk
        with self._write_lock:
            for key, metadata in self._key_metadata.items():
                if key in data:
                    self._write_to_cache(item, key, data[key], metadata)

        self._is_item_cached[item] = True
        return data

    def _write_to_cache(self, item: int, key: str, value: np.ndarray, metadata: CacheKeyMetadata) -> None:
        """Write a single array to the cache.

        Writes to a per-process/thread temporary file and atomically renames it
        into place. DataLoader workers are separate *processes*, so the
        ``threading.Lock`` gives no cross-process protection; the atomic rename
        ensures readers never observe a half-written file even if two workers
        race on the same item.
        """
        filepath = self._get_cache_filepath(item, key, metadata)

        # Skip if already cached (fast path)
        if filepath.exists():
            return

        # Keep the real suffix at the end so cv2 picks the right encoder.
        tmp_path = filepath.with_name(
            f".{filepath.stem}.{os.getpid()}.{threading.get_ident()}{filepath.suffix}"
        )

        try:
            if metadata.is_native_image:
                save_image(value, tmp_path)
            elif metadata.can_store_as_image:
                converted = convert_to_image(value, value.dtype)
                save_image(converted, tmp_path)
            else:
                # np.save appends .npy; write to the exact tmp path instead.
                with open(tmp_path, "wb") as f:
                    np.save(f, value)
            os.replace(tmp_path, filepath)
        except Exception as e:
            logger.error(f"Failed to cache {filepath}: {e}")
            # Clean up partial temp file if it exists
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise

    def remap(self, old_key: str, new_key: str) -> None:
        """
        Rename a key in the cache.

        Note: This only updates in-memory tracking. Cached files on disk
        are not renamed.

        Args:
            old_key: Current key name
            new_key: New key name
        """
        if old_key in self._key_metadata:
            self._key_metadata[new_key] = self._key_metadata.pop(old_key)

        if old_key in self._in_memory_keys:
            self._in_memory_keys.remove(old_key)
            self._in_memory_keys.add(new_key)

    def clear_cache(self) -> None:
        """Delete all cached files."""
        if self._root_cache_folder is None:
            return

        import shutil

        if self._root_cache_folder.exists():
            shutil.rmtree(self._root_cache_folder)
            logger.info(f"Cleared cache at {self._root_cache_folder}")

        self._is_item_cached[:] = False
        self._folders_created = False

    def get_cache_size(self) -> int:
        """
        Get the total size of cached files in bytes.

        Returns:
            Total cache size in bytes
        """
        if self._root_cache_folder is None or not self._root_cache_folder.exists():
            return 0

        total_size = 0
        for filepath in self._root_cache_folder.rglob("*"):
            if filepath.is_file():
                total_size += filepath.stat().st_size

        return total_size

    def cleanup(self) -> None:
        """Release resources (does not delete cached files)."""
        self._key_metadata.clear()
        self._in_memory_keys.clear()
        self._root_cache_folder = None
        self._folders_created = False
        super().cleanup()
