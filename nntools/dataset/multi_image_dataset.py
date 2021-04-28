import glob

import cv2
import numpy as np

from nntools.dataset.image_tools import resize
from nntools.tracker import Log
from nntools.utils.io import read_image, path_leaf
from nntools.utils.misc import to_iterable
from .image_dataset import ImageDataset, supportedExtensions

NN_FILL_DOWNSAMPLE = '0'
NN_FILL_UPSAMPLE = '1'
MISSING_DATA_FLAG = '2'


class MultiImageDataset(ImageDataset):
    def __init__(self, img_url,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 sort_function=None,
                 use_cache=False,
                 filling_strategy=NN_FILL_DOWNSAMPLE):

        self.root_path = {k: to_iterable(path) for k, path in img_url.items()}
        self.filling_strategy = filling_strategy
        super(MultiImageDataset, self).__init__(shape=shape, keep_size_ratio=keep_size_ratio,
                                                recursive_loading=recursive_loading,
                                                sort_function=sort_function, use_cache=use_cache)

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

            list_common_file = []
            for k, img_filenames in imgs_filenames.items():
                list_common_file += img_filenames

            intersection = set(list_common_file)

            if self.filling_strategy == NN_FILL_DOWNSAMPLE:
                Log.warn("Downsampling the dataset to size %i" % min(list_lengths))

                for k in self.img_filepath.keys():
                    self.img_filepath[k] = np.asarray(
                        [img for img, filename in zip(self.img_filepath[k], imgs_filenames[k]) if filename in
                         intersection])
            elif self.filling_strategy == NN_FILL_UPSAMPLE:
                Log.warn("Upsampling missing labels to fit the dataset's size (%i)" % max(list_lengths))
                max_size = 0
                for k, list_file in imgs_filenames.items():
                    if len(list_file) > max_size:
                        max_size = len(list_file)
                        largest_list = list_file
                for k in self.img_filepath.keys():
                    root_k = []
                    for img_name in largest_list:
                        if img_name in imgs_filenames[k]:
                            root_k.append(self.img_filepath[k][imgs_filenames[k].index(img_name)])
                        else:
                            root_k.append(MISSING_DATA_FLAG)
                    self.img_filepath[k] = np.asarray(root_k)

        if self.sort_function is None and self.filling_strategy == NN_FILL_DOWNSAMPLE:
            for k in self.img_filepath.keys():
                img_argsort = np.argsort(self.img_filepath[k])
                self.img_filepath[k] = self.img_filepath[k][img_argsort]

        elif self.filling_strategy == NN_FILL_DOWNSAMPLE:
            for k in self.img_filepath.keys():
                img_argsort = np.argsort([self.sort_function(x) for x in imgs_filenames[k]])
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
                img = resize(image=img, shape=self.shape, keep_size_ratio=self.keep_size_ratio,
                             flag=cv2.INTER_NEAREST)
            inputs[k] = img

        return inputs
