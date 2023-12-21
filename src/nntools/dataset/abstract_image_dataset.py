import logging
import math
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field
from torch.utils.data import Dataset

from nntools import MISSING_DATA_FLAG, NN_FILL_UPSAMPLE
from nntools.dataset.image_tools import pad, resize
from nntools.dataset.utils import convert_dict_to_plottable
from nntools.utils.io import path_leaf, read_image
from nntools.utils.misc import identity, to_iterable
from nntools.utils.plotting import plot_images

from .tools import Composition

supportedExtensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".jp2", ".exr", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm"}
supportedExtensions.update({ext.upper() for ext in supportedExtensions})

plt.rcParams["image.cmap"] = "gray"


AllowedImreadFlags = Literal["cv2.IMREAD_UNCHANGED", "cv2.IMREAD_GRAYSCALE", "cv2.IMREAD_COLOR"]
AlloweInterpolationFlags = Literal[
    "cv2.INTER_NEAREST", "cv2.INTER_LINEAR", "cv2.INTER_CUBIC", "cv2.INTER_AREA", "cv2.INTER_LANCZOS4"
]


def shape_converter(shape: Union[int, Tuple[int, int], None]) -> Optional[Tuple[int, int]]:
    if shape is None:
        return None
    if isinstance(shape, int):
        return (shape, shape)
    return shape


@define
class AbstractImageDataset(Dataset, ABC):
    img_root: Union[Path, List[Path], Dict[str, Path], Dict[str, List[Path]]] = field(converter=to_iterable)
    shape: Optional[Union[int, Tuple[int, int]]] = field(default=None, converter=shape_converter)
    keep_size_ratio: bool = True
    extract_image_id_function: Callable[[str], str] = identity
    recursive_loading: bool = True
    use_cache: bool = False
    flag: AllowedImreadFlags = cv2.IMREAD_UNCHANGED
    return_indices: bool = False
    cmap_name: str = "jet_r"
    multiplicative_size_factor: float = 1
    return_tag: bool = False
    id: str = ""
    tag: Optional[Union[str, List[str]]] = None
    interpolation_flag: AlloweInterpolationFlags = cv2.INTER_LINEAR

    auto_pad: bool = field()

    @auto_pad.default
    def _auto_pad_default(self):
        if self.shape is None:
            return False
        return True

    @auto_pad.validator
    def _auto_pad_validator(self, attribute, value):
        if self.shape is None and value:
            raise ValueError("auto_pad cannot be True if shape is None")

    auto_resize: bool = field()

    @auto_resize.default
    def _auto_resize_default(self):
        if self.shape is None:
            return False
        return True

    @auto_resize.validator
    def _auto_resize_validator(self, attribute, value):
        if self.shape is None and value:
            raise ValueError("auto_resize cannot be True if shape is None")

    _precache_composer: Optional[Composition] = field(default=None)
    _composer = None

    def __attrs_post_init__(self):
        self.ignore_keys = []
        self.img_filepath = {"image": []}
        self.gts = {}
        self.shared_arrays = {}
        self.cache_with_shared_array = True
        self.shm = None
        self.cache_initialized = False
        self.cache_filled = False
        self._is_first_process = False
        self.list_files(self.recursive_loading)

    def __len__(self):
        return int(self.multiplicative_size_factor * self.real_length)

    @property
    def real_length(self):
        return len(self.img_filepath["image"])

    @property
    def filenames(self):
        return {k: [f.name for f in v] for k, v in self.img_filepath.items()}

    @property
    def gt_filenames(self):
        return {k: [f.name for f in v] for k, v in self.gts.items()}

    @property
    def composer(self):
        return self._composer

    @composer.setter
    def composer(self, comp: Composition):
        self._composer = comp

    @abstractmethod
    def list_files(self, recursive):
        pass

    def load_image(self, item: int):
        inputs = {}
        for k, file_list in self.img_filepath.items():
            filepath = file_list[item]
            if filepath == MISSING_DATA_FLAG and self.filling_strategy == NN_FILL_UPSAMPLE:
                img = np.zeros(self.shape, dtype=np.uint8)
            else:
                img = read_image(filepath, flag=self.flag)
                img = self.resize_and_pad(image=img, interpolation=self.interpolation_flag)
            inputs[k] = img
        return inputs

    def precompose_data(self, data):
        if self.composer:
            return self.composer.precache_call(**data)
        else:
            return data

    def resize_and_pad(self, image, interpolation=cv2.INTER_CUBIC):
        if self.auto_resize:
            image = resize(image=image, shape=self.shape, keep_size_ratio=self.keep_size_ratio, flag=interpolation)
        if self.auto_pad:
            image = pad(image=image, shape=self.shape)
        return image

    def multiply_size(self, factor: float):
        self.multiplicative_size_factor = factor

    def __del__(self):
        if self.use_cache and self.cache_initialized and self._is_first_process:
            for shm in self.shms:
                shm.close()
            if self._is_first_process:
                for shm in self.shms:
                    shm.unlink()
            self.cache_initialized = False
            self._is_first_process = False

    def init_cache(self):
        self.use_cache = True
        if self.cache_initialized:
            return
        if not self.auto_resize and not self.auto_pad:
            logging.warning(
                "You are using a cache with auto_resize and auto_pad set to False.\
                    Make sure all your images are the same size"
            )

        arrays = self.load_image(0)  # Taking the first element
        arrays = self.precompose_data(arrays)

        shared_arrays = dict()
        nb_samples = self.real_length
        self.shms = []
        # Keep reference to all shm avoid the call from the garbage collector which pointer to buffer error
        if self.cache_with_shared_array:
            try:
                shm = shared_memory.SharedMemory(name=f"nntools_{self.id}_is_item_cached", size=nb_samples, create=True)
                self._is_first_process = True
            except FileExistsError:
                shm = shared_memory.SharedMemory(
                    name=f"nntools_{self.id}_is_item_cached", size=nb_samples, create=False
                )
                self._is_first_process = False

            self.shms.append(shm)
            self._cache_items = np.frombuffer(buffer=shm.buf, dtype=bool)
            self._cache_items[:] = 0

        for key, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                shared_arrays[key] = np.ndarray(nb_samples, dtype=type(arr))
                continue

            memory_shape = (nb_samples, *arr.shape)
            if self.cache_with_shared_array:
                try:
                    shm = shared_memory.SharedMemory(
                        name=f"nntools_{key}_{self.id}", size=arr.nbytes * nb_samples, create=True
                    )
                    logging.info(f"Creating shared memory, {mp.current_process().name}")
                    logging.debug(f"nntools_{key}_{self.id}: size: {shm.buf.nbytes} ({memory_shape})")
                except FileExistsError:
                    shm = shared_memory.SharedMemory(
                        name=f"nntools_{key}_{self.id}", size=arr.nbytes * nb_samples, create=False
                    )
                    logging.info(f"Assessing existing shared memory {mp.current_process().name}")

                self.shms.append(shm)

                shared_array = np.frombuffer(buffer=shm.buf, dtype=arr.dtype).reshape(memory_shape)

                shared_array[:] = 0
                # The initialization with 0 is not needed.
                # However, it's a good way to check if the shared memory is correctly initialized
                # It checks if there is enough space in dev/shm

                shared_arrays[key] = shared_array
            else:
                shared_arrays[key] = np.zeros(memory_shape, dtype=arr.dtype)

        self.shared_arrays = shared_arrays
        self.cache_initialized = True

    def load_array(self, item: int):
        if not self.use_cache:
            data = self.load_image(item)
            return self.precompose_data(data)
        else:
            if not self.cache_initialized:
                self.init_cache()

            if self._cache_items[item]:
                return {k: v[item] for k, v in self.shared_arrays.items()}

            arrays = self.load_image(item)
            arrays = self.precompose_data(arrays)
            for k, array in arrays.items():
                if array.ndim == 2:
                    self.shared_arrays[k][item, :, :] = array[:, :]
                else:
                    self.shared_arrays[k][item, :, :, :] = array[:, :, :]
            self._cache_items[item] = True
            return arrays

    def columns(self):
        return (self.img_filepath.keys(), self.gts.keys())

    def remap(self, old_key: str, new_key: str):
        dicts = [self.img_filepath, self.gts, self.shared_arrays]
        for d in dicts:
            if old_key in d.keys():
                d[new_key] = d.pop(old_key)

    def filename(self, items: List[int], key: str = "image"):
        items = np.asarray(items)
        filepaths = self.img_filepath[key][items]
        if isinstance(filepaths, list) or isinstance(filepaths, np.ndarray):
            return [os.path.basename(f) for f in filepaths]
        else:
            return os.path.basename(filepaths)

    def get_class_count(self, load: bool = True, save: bool = True):
        pass

    def transpose_img(self, img: np.ndarray):
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        elif img.ndim == 2:
            img = np.expand_dims(img, 0)
        return img

    def subset(self, indices: List[int]):
        for k, files in self.img_filepath.items():
            self.img_filepath[k] = files[indices]
        for k, files in self.gts.items():
            self.gts[k] = files[indices]

    def __getitem__(self, index: int, return_indices: bool = False, return_tag: bool = False):
        if abs(index) >= len(self):
            raise StopIteration
        if index >= self.real_length:
            index = int(index % self.real_length)

        inputs = self.load_array(index)
        if self.composer:
            outputs = self.composer.postcache_call(**inputs)
        else:
            outputs = inputs

        if self.return_indices or return_indices:
            outputs["index"] = index

        if self.tag and (self.return_tag or return_tag):
            if isinstance(self.tag, dict):
                for k, v in self.tag.items():
                    outputs[k] = v
            else:
                outputs["tag"] = self.tag
        outputs = self.filter_data(outputs)
        return outputs

    def filter_data(self, datadict: Dict[str, np.ndarray]):
        list_keys = list(datadict.keys())
        filtered_dict = {}
        for k in list_keys:
            if k not in self.ignore_keys:
                filtered_dict[k] = datadict[k]
        return filtered_dict

    def set_ignore_key(self, key):
        self.ignore_keys.append(key)

    def clean_filter(self):
        self.ignore_keys = []

    def plot(self, item: int, classes: Optional[List[str]] = None, fig_size: int = 1):
        arrays = self.__getitem__(item, return_indices=False)
        arrays = convert_dict_to_plottable(arrays)
        plot_images(arrays, self.cmap_name, classes=classes, fig_size=fig_size)

    def get_mosaic(
        self,
        n_items: int = 9,
        shuffle: bool = False,
        indexes: Optional[List[int]] = None,
        resolution: Tuple[int, int] = (512, 512),
        show: bool = False,
        fig_size: int = 1,
        save: Optional[bool] = None,
        add_labels: bool = False,
        n_row: Optional[int] = None,
        n_col: Optional[int] = None,
        n_classes: Optional[int] = None,
    ):
        if indexes is None:
            if shuffle:
                indexes = np.random.randint(0, len(self), n_items)
            else:
                indexes = np.arange(n_items)

        ref_dict = self.__getitem__(0, return_indices=False, return_tag=False)
        ref_dict = convert_dict_to_plottable(ref_dict)
        count_images = 0
        for k, v in ref_dict.items():
            if isinstance(v, np.ndarray) and not np.isscalar(v):
                count_images += 1
        if n_row is None and n_col is None:
            n_row = math.ceil(math.sqrt(n_items))
            n_col = math.ceil(n_items / n_row)
        if n_row * n_col < n_items:
            logging.warning("With %i columns, %i row(s), only %i items can be plotted" % (n_col, n_row, n_row * n_col))
            n_items = n_row * n_col
        pad = 50 if add_labels else 0
        cols = []

        for r in range(n_row):
            row = []
            for c in range(n_col):
                i = n_row * c + r
                if i >= n_items:
                    for n in range(count_images):
                        tmp = np.zeros((resolution[0] + pad, resolution[1], 3))
                        row.append(tmp)
                    continue
                index = indexes[i]
                data = self.__getitem__(index, return_indices=False, return_tag=False)
                data = convert_dict_to_plottable(data)

                for k, v in data.items():
                    if v.ndim == 3 and v.shape[-1] != 3:
                        v_tmp = np.argmax(v, axis=-1) + 1
                        v_tmp[v.max(axis=-1) == 0] = 0
                        v = v_tmp
                    if v.ndim == 3:
                        v = (v - v.min()) / (v.max() - v.min())
                    if v.ndim == 2:
                        n_classes = np.max(v) + 1 if n_classes is None else n_classes
                        if n_classes == 1:
                            n_classes = 2
                        cmap = plt.get_cmap(self.cmap_name, n_classes)
                        v = cmap(v)[:, :, :3]
                    if v.shape:
                        v = cv2.resize(v, resolution, cv2.INTER_NEAREST_EXACT)
                    if add_labels and v.shape:
                        v = np.pad(v, ((pad, 0), (0, 0), (0, 0)))
                        if k in self.gts:
                            text = self.gts[k][index]
                        elif k in self.img_filepath:
                            text = self.img_filepath[k][index]
                        else:
                            text = ""
                        text = os.path.basename(text)
                        font = cv2.FONT_HERSHEY_PLAIN
                        fontScale = 1.75
                        fontColor = (255, 255, 255)
                        lineType = 2

                        textsize = cv2.getTextSize(text, font, fontScale, lineType)[0]
                        textX = (v.shape[1] - textsize[0]) // 2
                        textY = (textsize[1] + pad) // 2

                        bottomLeftCornerOfText = textX, textY
                        cv2.putText(v, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                    if v.shape:
                        row.append(v)

            rows = np.hstack(row)

            cols.append(rows)

        mosaic = np.vstack(cols)
        if show:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(mosaic)
            fig.set_size_inches(fig_size * 5 * count_images * n_col, 5 * n_row * fig_size)
            plt.axis("off")
            plt.tight_layout()
            fig.show()
        if save:
            assert isinstance(save, str)
            cv2.imwrite(save, (mosaic * 255)[:, :, ::-1])

        return mosaic
