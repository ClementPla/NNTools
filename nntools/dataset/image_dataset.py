import ctypes
import math
import multiprocessing as mp
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from torch.utils.data import Dataset

from nntools.dataset.image_tools import resize
from nntools.utils.io import read_image
from nntools.utils.plotting import plot_images
from nntools.utils.misc import to_iterable, identity
from nntools.tracker.logger import Log
supportedExtensions = ["jpg", "jpeg", "png", "tiff", "tif", "jp2", "exr", "pbm", "pgm", "ppm", "pxm", "pnm"]
plt.rcParams['image.cmap'] = 'gray'


class ImageDataset(Dataset):
    def __init__(self, img_url=None,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 extract_image_id_function=None,
                 use_cache=False):

        if extract_image_id_function is None:
            self.extract_image_id_function = identity
        else:
            self.extract_image_id_function = extract_image_id_function
        if img_url is not None:
            self.path_img = to_iterable(img_url)
        self.composer = None
        self.keep_size_ratio = keep_size_ratio
        self.shape = tuple(shape)
        self.recursive_loading = recursive_loading

        self.img_filepath = {'image': []}
        self.gts = {}
        self.shared_arrays = {}

        self.auto_resize = True
        self.return_indices = False
        self.list_files(recursive_loading)

        self.use_cache = use_cache
        self.cmap_name = 'jet_r'

        self.multiplicative_size_factor = 1
        if self.use_cache:
            self.cache()

        self.tag = None
        self.return_tag = False

        self.ignore_keys = []

    def __len__(self):
        return int(self.multiplicative_size_factor * self.real_length)

    @property
    def real_length(self):
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

    def multiply_size(self, factor):
        assert factor > 1
        self.multiplicative_size_factor = factor

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

    def columns(self):
        return (self.img_filepath.keys(), self.gts.keys())

    def remap(self, old_key, new_key):
        dicts = [self.img_filepath, self.gts, self.shared_arrays]
        for d in dicts:
            if old_key in d.keys():
                d[new_key] = d.pop(old_key)

    def filename(self, items, col='image'):
        items = np.asarray(items)
        filepaths = self.img_filepath[col][items]
        if isinstance(filepaths, list) or isinstance(filepaths, np.ndarray):
            return [os.path.basename(f) for f in filepaths]
        else:
            return os.path.basename(filepaths)

    def set_composition(self, composer):
        self.composer = composer

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

    def __getitem__(self, index, torch_cast=True, transpose_img=True, return_indices=True,
                    return_tag=True):
        if abs(index) >= len(self):
            raise StopIteration
        if index >= self.real_length:
            index = int(index % self.real_length)

        inputs = self.load_array(index)

        if self.composer:
            outputs = self.composer(**inputs)
        else:
            outputs = inputs

        for k, item in outputs.items():
            if not isinstance(item, np.ndarray):
                item = np.asarray(item)
            if item.ndim == 3 and transpose_img:
                item = self.transpose_img(item)  # HWN to NHW
            if torch_cast:
                item = torch.from_numpy(item.copy())
                if isinstance(item, torch.ByteTensor):
                    item = item.long()
            outputs[k] = item

        if self.return_indices and return_indices:
            outputs['index'] = index

        if self.tag and (self.return_tag and return_tag):
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
        arrays = self.__getitem__(item, torch_cast=False, transpose_img=False, return_indices=False)
        plot_images(arrays, self.cmap_name, classes=classes, fig_size=fig_size)
        
    def get_mosaic(self, n_items=9, shuffle=False, indexes=None, resolution=(512, 512), show=False, fig_size=1,
                   save=None, add_labels=False,
                   n_row=None, n_col=None):

        if indexes is None:
            if shuffle:
                indexes = np.random.randint(0, len(self), n_items)
            else:
                indexes = np.arange(n_items)

        ref_dict = self.__getitem__(0, torch_cast=False, transpose_img=False, return_indices=False, return_tag=False)
        count_images = 0
        for k, v in ref_dict.items():
            if isinstance(v, np.ndarray) and not np.isscalar(v):
                count_images += 1
        if n_row is None and n_col is None:
            n_row = math.ceil(math.sqrt(n_items))
            n_col = math.ceil(n_items / n_row)
        if n_row*n_col < n_items:
            Log.warn("With %i columns, %i row(s), only %i items can be plotted" % (n_col, n_row, n_row*n_col))
            n_items = n_row*n_col
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
                data = self.__getitem__(index, torch_cast=False, transpose_img=False, return_indices=False,
                                        return_tag=False)

                for k, v in data.items():
                    if v.ndim == 3 and v.shape[-1] > 3:
                        v_tmp = np.argmax(v, axis=-1) + 1
                        v_tmp[v.max(axis=-1) == 0] = 0
                        v = v_tmp
                    if v.ndim == 3:
                        v = (v - v.min()) / (v.max() - v.min())
                    if v.ndim == 2:
                        n_classes = np.max(v) + 1
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
                            text = ''
                        text = os.path.basename(text)
                        font = cv2.FONT_HERSHEY_PLAIN
                        fontScale = 1.75
                        fontColor = (255, 255, 255)
                        lineType = 2

                        textsize = cv2.getTextSize(text, font, fontScale, lineType)[0]
                        textX = (v.shape[1] - textsize[0]) // 2
                        textY = (textsize[1] + pad) // 2

                        bottomLeftCornerOfText = textX, textY
                        cv2.putText(v, text,
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                    if v.shape:
                        row.append(v)

            rows = np.hstack(row)

            cols.append(rows)

        mosaic = np.vstack(cols)
        if show:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(mosaic)
            fig.set_size_inches(fig_size * 5 * count_images * n_col, 5 * n_row * fig_size)
            fig.show()
        if save:
            assert isinstance(save, str)
            cv2.imwrite(save, (mosaic * 255)[:, :, ::-1])

        return mosaic
