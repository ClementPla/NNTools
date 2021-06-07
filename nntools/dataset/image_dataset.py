import os

import numpy as np
import torch
from torch.utils.data import Dataset

from nntools.dataset.image_tools import resize
from nntools.utils.io import read_image
from nntools.utils.misc import to_iterable, identity

supportedExtensions = ["jpg", "jpeg", "png", "tiff", "tif", "jp2", "exr", "pbm", "pgm", "ppm", "pxm", "pnm"]
import multiprocessing as mp
import ctypes
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

plt.rcParams['image.cmap'] = 'gray'
import math


class ImageDataset(Dataset):
    def __init__(self, img_url=None,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 sort_function=None,
                 use_cache=False):

        if sort_function is None:
            self.sort_function = identity
        else:
            self.sort_function = sort_function
        if img_url is not None:
            self.path_img = to_iterable(img_url)
        self.composer = None
        self.keep_size_ratio = keep_size_ratio
        self.shape = tuple(shape)
        self.recursive_loading = recursive_loading
        self.img_filepath = {'image': []}
        self.gts = {}
        self.auto_resize = True
        self.return_indices = False
        self.list_files(recursive_loading)
        self.use_cache = use_cache
        self.cmap_name = 'jet_r'

        if self.use_cache:
            self.cache()

    def __len__(self):
        return len(self.img_filepath['image'])

    def list_files(self, recursive):
        pass

    def read_sharred_array(self, item):
        return {k: self.shared_arrays[k][item] for k in self.shared_arrays}

    def load_image(self, item):
        filepath = self.img_filepath['image'][item]
        img = read_image(filepath)
        if self.auto_resize:
            img = resize(image=img, shape=self.shape,
                         keep_size_ratio=self.keep_size_ratio)
        return {'image': img}

    def init_cache(self):
        self.use_cache = False
        arrays = self.load_array(0)  # Taking the first element
        shared_arrays = {}
        nb_samples = len(self)
        for key, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                shared_arrays[key] = arr
                continue
            if arr.ndim == 2:
                h, w = arr.shape
                c = 1
            else:
                h, w, c = arr.shape
            shared_array_base = mp.Array(ctypes.c_uint8, nb_samples * c * h * w, lock=True)
            with shared_array_base.get_lock():
                shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
                if c > 1:
                    shared_array = shared_array.reshape(nb_samples, h, w, c)
                else:
                    shared_array = shared_array.reshape(nb_samples, h, w)
                shared_array[0] = arr
                shared_arrays[key] = shared_array
        self.shared_arrays = shared_arrays

    def cache(self):
        self.use_cache = False
        self.init_cache()
        print('Caching dataset...')
        for item in tqdm.tqdm(range(1, len(self))):
            arrays = self.load_array(item)
            for k, arr in arrays.items():
                self.shared_arrays[k][item] = arr
        self.use_cache = True

    def load_array(self, item):
        if self.use_cache:
            return self.read_sharred_array(item)
        else:
            return self.load_image(item)

    def filename(self, items):
        items = np.asarray(items)
        filepaths = self.img_filepath['image'][items]
        if isinstance(filepaths, list) or isinstance(filepaths, np.ndarray):
            return [os.path.basename(f) for f in filepaths]
        else:
            return os.path.basename(filepaths)

    def set_composition(self, composer):
        self.composer = composer

    def get_class_count(self):
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

    def __getitem__(self, index, torch_cast=True, transpose_img=True, return_indices=True):
        inputs = self.load_array(index)
        if self.composer:
            outputs = self.composer(**inputs)
        else:
            outputs = inputs

        if transpose_img:
            for k, item in outputs.items():
                if not isinstance(item, np.ndarray):
                    item = np.asarray(item)
                if item.ndim == 3 and transpose_img:
                    item = self.transpose_img(item) # HWN to NHW
                if torch_cast:
                    item = torch.from_numpy(item)
                    if isinstance(item, torch.ByteTensor):
                        item = item.long()
                outputs[k] = item

        if self.return_indices and return_indices:
            outputs['indice'] = index
        return outputs

    def plot(self, item, classes=None):
        arrays = self.__getitem__(item, torch_cast=False, transpose_img=False, return_indices=False)

        arrays = [(k, v) for k, v in arrays.items() if isinstance(v, np.ndarray)]
        nb_plots = len(arrays)
        row, col = int(math.ceil(nb_plots / 2)), 2

        fig, ax = plt.subplots(row, col)
        if row == 1:
            ax = [ax]
        fig.set_size_inches(10, 5 * row)

        for i in range(row):
            for j in range(col):
                ax[i][j].set_axis_off()

                if j + i * col >= len(arrays):
                    ax[i][j].imshow(np.zeros_like(arr))
                else:
                    name, arr = arrays[j + i * col]
                    n_classes = np.max(arr)+1
                    if n_classes == 0:
                        n_classes = 2
                    cmap = cm.get_cmap(self.cmap_name, n_classes)
                    ax[i][j].set_title(name)

                    if arr.ndim == 3:
                        arr = (arr - np.min(arr))/(np.max(arr) - np.min(arr))
                        ax[i][j].imshow(np.squeeze(arr), cmap='gray')
                    elif arr.ndim == 2:
                        ax[i][j].imshow(np.squeeze(arr), cmap=cmap)
                        divider = make_axes_locatable(ax[i][j])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        cax.imshow(np.expand_dims(np.arange(n_classes), 0).transpose((1, 0)), aspect='auto', cmap=cmap)
                        cax.yaxis.set_label_position("right")
                        cax.yaxis.tick_right()
                        if classes is not None:
                            cax.set_yticklabels(labels=classes)
                        cax.yaxis.set_ticks(np.arange(n_classes))
                        cax.get_xaxis().set_visible(False)

        fig.tight_layout()
        fig.show()
