import logging
from pathlib import Path

import cv2
import numpy as np

from nntools.dataset.cache.abstract_cache import AbstractCache
from nntools.utils.io import read_image, save_image
from nntools.utils.misc import can_be_stored_as_image, convert_to_image, is_image, revert_image_to_original_dtype


class Metadata:
    def __init__(self, cache_folder, is_image, can_be_stored_as_image):
        self.cache_folder = cache_folder
        self.is_image = is_image
        self.can_be_stored_as_image = can_be_stored_as_image


class DiskCache(AbstractCache):
    def __init__(self, dataset, cache_dir: Path = "") -> None:
        super().__init__(dataset)

        self.cache_folders = {}
        self.in_memory_items = []
        self.shms = []
        self.root_cache_folder = None
        self.needs_filling = False
        self.cache_dir = cache_dir

    def init_cache(self) -> None:
        if self.is_initialized:
            return
        self.root_cache_folder = self.get_cache_folder()
        self.root_cache_folder = self.root_cache_folder

        arrays = self.d.get_precache_data(0)  # Initialization is based on the first element of the dataset

        self.init_non_shared_items_tracking()

        for k, v in arrays.items():
            if not isinstance(v, np.ndarray):
                assert k in self.d.gts, f"Key {k} not found in dataset ground truths. \
                As it is not a numpy array, it can't be cached (for now)."
                self.in_memory_items.append(k)
            else:
                k_cache_folder = self.root_cache_folder / k
                metadata = Metadata(k_cache_folder, is_image(v), can_be_stored_as_image(v))
                self.cache_folders[k] = metadata

        self.is_item_cached[:] = False
        for k, v in self.cache_folders.items():
            if not v.cache_folder.exists():
                v.cache_folder.mkdir(parents=True, exist_ok=False)
                logging.info(f"Creating cache folder {self.root_cache_folder}.")

        self.check_if_filling_is_needed()
        self.is_initialized = True

    def check_if_filling_is_needed(self):
        for i in range(self.nb_samples):
            self.is_item_cached[i] = self.check_cache(i)

    def __getitem__(self, item):
        if self.is_item_cached[item] or self.check_cache(item):
            return self.get_cached_item(item)
        else:
            return self.cache_item(item)

    def get_cached_item(self, item):
        data = {k: self.d.gts[k][item] for k in self.in_memory_items}
        for k, metadata in self.cache_folders.items():
            if k in self.d.on_disk_keys:
                name = self.d.filename(item, k)
            else:
                name = self.d.filename(item)  # Take the filename of the image
                name = Path(name).with_suffix(".png")

            if metadata.is_image:
                data[k] = read_image(metadata.cache_folder / name, cv2.IMREAD_UNCHANGED)
            elif metadata.can_be_stored_as_image:
                cached = read_image((metadata.cache_folder / name).with_suffix(".png"), cv2.IMREAD_UNCHANGED)
                cached = np.ascontiguousarray(cached)
                data[k] = revert_image_to_original_dtype(cached, metadata.can_be_stored_as_image)
            else:
                data[k] = np.load((metadata.cache_folder / name).with_suffix(".npy"))
        return data

    def check_cache(self, item):
        for k, metadata in self.cache_folders.items():
            if k in self.d.on_disk_keys:
                name = self.d.filename(item, k)
            else:
                name = self.d.filename(item)  # Take the filename of the image
                name = Path(name).with_suffix(".png")

            filepath = metadata.cache_folder / Path(name)

            if metadata.is_image:
                if not filepath.exists():
                    return False
            elif metadata.can_be_stored_as_image:
                if not (metadata.cache_folder / Path(name).with_suffix(".png")).exists():
                    return False
            else:
                if not (metadata.cache_folder / Path(name).with_suffix(".npy")).exists():
                    return False
        return True

    def cache_item(self, item):
        if self.check_cache(item):
            return self.get_cached_item(item)
        arrays = self.d.read_from_disk(item)
        arrays = self.d.precompose_data(arrays)
        for k, v in arrays.items():
            if k in self.in_memory_items:
                continue
            else:
                self.cache_to_disk(k, v, item)

        self.is_item_cached[item] = True
        return arrays

    def cache_to_disk(self, key, value, item):
        if key in self.d.on_disk_keys:
            name = Path(self.d.filename(item, key))
        else:
            name = self.d.filename(item)
            name = Path(name).with_suffix(".png")

        filepath: Path = self.cache_folders[key].cache_folder / name

        if self.cache_folders[key].is_image:
            save_image(value, filepath)
        elif self.cache_folders[key].can_be_stored_as_image:
            item = convert_to_image(value, value.dtype)
            save_image(item, filepath.with_suffix(".png"))
        else:
            np.save(filepath.with_suffix(".npy"), value)

    def get_cache_folder(self) -> Path:
        root_img = Path(self.d.img_filepath["image"][0]).parent
        folder_name = root_img.name
        cache_folder = root_img.parent / f".{folder_name}_cache" / self.cache_dir / self.d.id / self.d.composer.id
        return cache_folder

    def remap(self, old_key: str, new_key: str):
        for k, v in self.cache_folders.items():
            if k == old_key:
                self.cache_folders[new_key] = v
                del self.cache_folders[k]
        if old_key in self.in_memory_items:
            self.in_memory_items.remove(old_key)

        self.in_memory_items.append(new_key)
