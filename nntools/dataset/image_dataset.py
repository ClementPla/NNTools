import glob

import cv2
import numpy as np
from nntools import NN_FILL_UPSAMPLE, NN_FILL_DOWNSAMPLE, MISSING_DATA_FLAG
from nntools.dataset.image_tools import resize
from nntools.tracker import Log
from nntools.utils.io import read_image, path_leaf
from nntools.utils.misc import to_iterable

from .abstract_image_dataset import AbstractImageDataset, supportedExtensions


class MultiImageDataset(AbstractImageDataset):
    def __init__(self, img_url,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 extract_image_id_function=None,
                 use_cache=False,
                 filling_strategy=NN_FILL_DOWNSAMPLE, flag=cv2.IMREAD_COLOR):

        self.root_path = {k: to_iterable(path) for k, path in img_url.items()}
        self.filling_strategy = filling_strategy
        super(MultiImageDataset, self).__init__(shape=shape, keep_size_ratio=keep_size_ratio,
                                                recursive_loading=recursive_loading,
                                                extract_image_id_function=extract_image_id_function,
                                                use_cache=use_cache, flag=flag)

    def match_images_number_per_folder(self, filenames_per_folder):

        list_lengths = [len(img_filenames) for img_filenames in filenames_per_folder.values()]
        if self.filling_strategy == NN_FILL_DOWNSAMPLE:
            Log.warn("Downsampling the dataset to size %i" % min(list_lengths))
            smallest_list = sorted(filenames_per_folder.values(), key=lambda x: len(x))[0]

            for k in self.img_filepath.keys():
                self.img_filepath[k] = np.asarray(
                    [img for img, filename in zip(self.img_filepath[k], filenames_per_folder[k]) if filename in
                     smallest_list])
        elif self.filling_strategy == NN_FILL_UPSAMPLE:
            Log.warn("Upsampling missing labels to fit the dataset's size (%i)" % max(list_lengths))

            largest_list = sorted(filenames_per_folder.values(), key=lambda x: len(x))[-1]
            for k in self.img_filepath.keys():
                root_k = []
                for img_name in largest_list:
                    if img_name in filenames_per_folder[k]:
                        root_k.append(self.img_filepath[k][filenames_per_folder[k].index(img_name)])
                    else:
                        root_k.append(MISSING_DATA_FLAG)
                self.img_filepath[k] = np.asarray(root_k)

    def list_files(self, recursive):
        self.img_filepath = {k: [] for k in self.root_path.keys()}

        for extension in supportedExtensions:
            prefix = "**/*." if recursive else "*."
            for root_label, paths in self.root_path.items():
                for path in paths:
                    self.img_filepath[root_label].extend(glob.glob(path + prefix + extension, recursive=recursive))

        for k, filepaths in self.img_filepath.items():
            self.img_filepath[k] = np.asarray(filepaths)

        """
        Sorting files
        """
        imgs_filenames = {}
        for k, files_list in self.img_filepath.items():
            imgs_filenames[k] = [path_leaf(file).split('.')[0] for file in files_list]

        list_lengths = [len(img_filenames) for img_filenames in imgs_filenames.values()]

        all_equal = all(elem == list_lengths[0] for elem in list_lengths)
        if not all_equal:
            Log.warn("Mismatch between the size of the different input folders (smaller %i, longer %i)" % (min(
                list_lengths), max(list_lengths)))
            self.match_images_number_per_folder(imgs_filenames)

        if self.filling_strategy == NN_FILL_DOWNSAMPLE:
            for k in self.img_filepath.keys():
                img_argsort = np.argsort([self.extract_image_id_function(x) for x in imgs_filenames[k]])
                self.img_filepath[k] = self.img_filepath[k][img_argsort]

    def __len__(self):
        if self.filling_strategy == NN_FILL_DOWNSAMPLE:
            return min([len(filepaths) for filepaths in self.img_filepath.values()])
        elif self.filling_strategy == NN_FILL_UPSAMPLE:
            return max([len(filepaths) for filepaths in self.img_filepath.values()])

    def load_image(self, item):
        inputs = {}
        for k, file_list in self.img_filepath.items():
            filepath = file_list[item]
            if filepath == MISSING_DATA_FLAG and self.filling_strategy == NN_FILL_UPSAMPLE:
                img = np.zeros(self.shape, dtype=np.uint8)
            else:
                img = read_image(filepath)
                if self.auto_resize:
                    img = resize(image=img, shape=self.shape, keep_size_ratio=self.keep_size_ratio,
                                 flag=cv2.INTER_CUBIC)
            inputs[k] = img

        return inputs


class ImageDataset(MultiImageDataset):
    def __init__(self, img_url,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 extract_image_id_function=None,
                 use_cache=False,
                 filling_strategy=NN_FILL_DOWNSAMPLE):
        super(ImageDataset, self).__init__(img_url={'image': img_url},
                                           shape=shape,
                                           keep_size_ratio=keep_size_ratio,
                                           recursive_loading=recursive_loading,
                                           extract_image_id_function=extract_image_id_function,
                                           use_cache=use_cache,
                                           filling_strategy=filling_strategy)
