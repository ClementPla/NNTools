from torch.utils.data import Dataset
import numpy as np
import os
from nntools.dataset.image_tools import resize
from nntools.utils.io import load_image

supportedExtensions = ["jpg", "jpeg", "png", "tiff", "tif", "jp2", "exr", "pbm", "pgm", "ppm", "pxm", "pnm"]


class ImageDataset(Dataset):
    def __init__(self, img_url,
                 shape=None,
                 keep_size_ratio=True,
                 recursive_loading=True):

        self.path_img = img_url
        self.composer = None
        self.keep_size_ratio = keep_size_ratio
        self.shape = tuple(shape)
        self.recursive_loading = recursive_loading
        self.img_filepath = []
        self.gts = []

        self.return_indices = False
        self.list_files(recursive_loading)

    def __len__(self):
        return len(self.img_filepath)

    def list_files(self, recursive):
        pass

    def load_image(self, item):
        filepath = self.img_filepath[item]
        img = load_image(filepath)
        img = resize(image=img, shape=self.shape,
                     keep_size_ratio=self.keep_size_ratio)
        return img

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


