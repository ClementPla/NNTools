import os
import torch
import numpy as np
from torch.utils.data import Dataset


from nntools.utils.misc import to_iterable
from nntools.dataset.image_tools import resize
from nntools.utils.io import read_image

supportedExtensions = ["jpg", "jpeg", "png", "tiff", "tif", "jp2", "exr", "pbm", "pgm", "ppm", "pxm", "pnm"]
import multiprocessing as mp
import ctypes
import tqdm


class ImageDataset(Dataset):
    def __init__(self, img_url,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 sort_function=None,
                 use_cache=False):
        self.sort_function = sort_function
        self.path_img = to_iterable(img_url)
        self.composer = None
        self.keep_size_ratio = keep_size_ratio
        self.shape = tuple(shape)
        self.recursive_loading = recursive_loading
        self.img_filepath = []
        self.gts = []
        self.auto_resize = True
        self.return_indices = False
        self.list_files(recursive_loading)
        self.use_cache = use_cache

        if self.use_cache:
            self.cache()

    def __len__(self):
        return len(self.img_filepath)

    def list_files(self, recursive):
        pass

    def read_sharred_array(self, item):
        return {k : self.shared_arrays[k][item] for k in self.shared_arrays}

    def load_image(self, item):
        filepath = self.img_filepath[item]
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
        filepaths = self.img_filepath[items]
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
        self.img_filepath = self.img_filepath[indices]
        self.gts = self.gts[indices]

    def __getitem__(self, item):
        inputs = self.load_array(item)
        if self.composer:
            outputs = self.composer(**inputs)
        else:
            outputs = inputs

        outputs['image'] = self.transpose_img(outputs['image'])
        for k in outputs:
            outputs[k] = torch.from_numpy(outputs[k])
        if self.return_indices:
            outputs['indice'] = item
        return outputs
