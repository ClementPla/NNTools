import logging
import math
import multiprocessing as mp

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import shared_memory

from nntools import MISSING_DATA_FLAG, NN_FILL_UPSAMPLE
from nntools.dataset.image_tools import pad, resize
from nntools.utils.io import path_leaf, read_image
from nntools.utils.misc import identity, to_iterable
from nntools.utils.plotting import plot_images
from torch.utils.data import get_worker_info
from .tools import Composition

supportedExtensions = ["jpg", "jpeg", "png", "tiff", "tif", "jp2", "exr", "pbm", "pgm", "ppm", "pxm", "pnm"]
supportedExtensions = supportedExtensions + [ext.upper() for ext in supportedExtensions]

plt.rcParams["image.cmap"] = "gray"


def convert_dict_to_plottable(dict_arrays):
    plotted_arrays = {}
    for k, v in dict_arrays.items():
        if isinstance(v, torch.Tensor):
            v = v.numpy()
            if v.ndim == 3:
                v = v.transpose((1, 2, 0))
        plotted_arrays[k] = v
    return plotted_arrays


class AbstractImageDataset(Dataset):
    def __init__(
        self,
        img_url=None,
        shape=None,
        keep_size_ratio=False,
        recursive_loading=True,
        extract_image_id_function=None,
        use_cache=False,
        auto_pad=True,
        flag=cv2.IMREAD_UNCHANGED,
    ):
        super().__init__()

        if extract_image_id_function is None:
            self.extract_image_id_function = identity
        else:
            self.extract_image_id_function = extract_image_id_function
        if img_url is not None:
            self.path_img = to_iterable(img_url)
        self._precache_composer = None
        self._composer = None
        self.keep_size_ratio = keep_size_ratio
        if isinstance(shape, int):
            shape = (shape, shape)
        self.shape = tuple(shape)
        self.recursive_loading = recursive_loading

        self.img_filepath = {"image": []}
        self.gts = {}
        self.shared_arrays = {}

        self.auto_resize = True
        self.return_indices = False
        self.list_files(recursive_loading)
        self.auto_pad = auto_pad

        self.use_cache = use_cache
        self.cmap_name = "jet_r"

        self.multiplicative_size_factor = 1

        self.tag = None
        self.return_tag = False

        self.ignore_keys = []
        self.flag = flag
        self._cache_initialized = False
        self.cache_with_shared_array = True 
        self.interpolation_flag = cv2.INTER_LINEAR
        self.shm = None
        
    def init_shared_values(self):
        self._cache_initialized = mp.Value('i', 0)
        self._cache_filled = mp.Value('i', 0) 
        
    def __len__(self):
        return int(self.multiplicative_size_factor * self.real_length)
        
    @property
    def real_length(self):
        return len(self.img_filepath["image"])

    @property
    def filenames(self):
        return {k: [path_leaf(f) for f in v] for k, v in self.img_filepath.items()}

    @property
    def gt_filenames(self):
        return {k: [path_leaf(f) for f in v] for k, v in self.gts.items()}
    
    @property
    def cache_initialized(self):
        if not hasattr(self, "_cache_initialized"):
            return False
        return bool(self._cache_initialized.value)
    
    @cache_initialized.setter
    def cache_initialized(self, cache_initialized):
        if cache_initialized:
            logging.info(f"Cache is marked as initialized")
        self._cache_initialized.value = int(cache_initialized)
    
    @property
    def cache_filled(self):
        return bool(self._cache_filled.value)
    
    @cache_filled.setter
    def cache_filled(self, cache_filled):
        if cache_filled:
            logging.info(f"Cache is marked as filled")
        self._cache_filled.value = int(cache_filled)
        
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

    def init_cache(self):
        if self.cache_initialized:
            return 
        self.init_shared_values()
        self.use_cache = True
        if not self.auto_resize and not self.auto_pad:
            logging.warning("You are using a cache with auto_resize and auto_pad set to False. Make sure all your images are the same size")
            
        arrays = self.load_image(0)  # Taking the first element
        arrays = self.precompose_data(arrays)
        
        shared_arrays = dict()
        nb_samples = self.real_length
        for key, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                shared_arrays[key] = np.ndarray(nb_samples, dtype=type(arr))
                continue
            if arr.ndim == 2:
                h, w = arr.shape
                c = 1
            else:
                h, w, c = arr.shape
            
            if self.cache_with_shared_array:
                try:
                    shm = shared_memory.SharedMemory(name=f'nntools_{key}_{str(self.id)}', size=arr.nbytes*nb_samples, create=True)
                    logging.info(f"Creating shared memory, {mp.current_process().name}")
                    logging.debug(f'nntools_{key}_{self.id.name}: size: {shm.buf.nbytes} ({nb_samples}x{h}x{w}x{c})')
                except FileExistsError:
                    shm = shared_memory.SharedMemory(name=f'nntools_{key}_{str(self.id)}',
                                                     size=arr.nbytes*nb_samples, create=False)
                    logging.info(f"Assessing existing shared memory {mp.current_process().name}")
                    logging.debug(f'nntools_{key}_{self.id.name}: size: {shm.buf.nbytes} ({nb_samples}x{h}x{w}x{c})')
                
                self.shm = shm
                if c>1:
                    shared_array = np.frombuffer(buffer=shm.buf, dtype=arr.dtype).reshape((nb_samples, h, w, c))
                else:
                    shared_array = np.frombuffer(buffer=shm.buf, dtype=arr.dtype).reshape((nb_samples, h, w))
                
                shared_arrays[key] = shared_array
            else:
                if c>1:
                    shared_arrays[key] = np.ndarray((nb_samples, h, w, c), dtype=arr.dtype)
                else:
                    shared_arrays[key] = np.ndarray((nb_samples, h, w), dtype=arr.dtype)
                        
        self.shared_arrays = shared_arrays
        
    def load_array(self, item):
        if not self.use_cache:
            data = self.load_image(item)
            return self.precompose_data(data)
        else:
            if not self.cache_initialized:
                self.init_cache()
            if not self.cache_filled:
                arrays = self.load_image(item)
                arrays = self.precompose_data(arrays)
                for k, array in arrays.items():
                    if array.ndim == 2:                    
                        self.shared_arrays[k][item, :, :] = array[:, :]
                    else:
                        self.shared_arrays[k][item, :, :, :] = array[:, :, :]
                return arrays
            else:
                return {k: v[item] for k, v in self.shared_arrays.items()}

    def columns(self):
        return (self.img_filepath.keys(), self.gts.keys())

    def remap(self, old_key, new_key):
        dicts = [self.img_filepath, self.gts, self.shared_arrays]
        for d in dicts:
            if old_key in d.keys():
                d[new_key] = d.pop(old_key)

    def filename(self, items, col="image"):
        items = np.asarray(items)
        filepaths = self.img_filepath[col][items]
        if isinstance(filepaths, list) or isinstance(filepaths, np.ndarray):
            return [os.path.basename(f) for f in filepaths]
        else:
            return os.path.basename(filepaths)

    @property
    def composer(self):
        return self._composer

    @composer.setter
    def composer(self, comp: Composition):
        self._composer = comp

    def get_class_count(self, load=True, save=True):
        pass

    def transpose_img(self, img):
        if img.ndim == 3:
            img = img.transpose(2, 0, 1)
        elif img.ndim == 2:
            img = np.expand_dims(img, 0)
        return img

    def subset(self, indices):
        for k, files in self.img_filepath.items():
            self.img_filepath[k] = files[indices]
        for k, files in self.gts.items():
            self.gts[k] = files[indices]

    def __getitem__(self, index, return_indices=False, return_tag=False):
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

    def filter_data(self, datadict):
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

    def plot(self, item, classes=None, fig_size=1):
        arrays = self.__getitem__(item, return_indices=False)
        arrays = convert_dict_to_plottable(arrays)
        plot_images(arrays, self.cmap_name, classes=classes, fig_size=fig_size)

    def get_mosaic(
        self,
        n_items=9,
        shuffle=False,
        indexes=None,
        resolution=(512, 512),
        show=False,
        fig_size=1,
        save=None,
        add_labels=False,
        n_row=None,
        n_col=None,
        n_classes=None,
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
