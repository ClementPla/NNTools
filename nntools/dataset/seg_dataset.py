import glob

import cv2
import numpy as np
from nntools.dataset.image_tools import resize
from nntools.tracker import Log
from nntools.utils.io import read_image, path_leaf
from nntools.utils.misc import to_iterable

from .image_dataset import ImageDataset, supportedExtensions
from nntools import NN_FILL_DOWNSAMPLE, NN_FILL_UPSAMPLE, MISSING_DATA_FLAG


class SegmentationDataset(ImageDataset):
    def __init__(self, img_url,
                 mask_url=None,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 n_classes=None,
                 extract_image_id_function=None,
                 use_cache=False,
                 filling_strategy=NN_FILL_UPSAMPLE):

        if mask_url is None or mask_url == '':
            self.path_masks = None
        elif not isinstance(mask_url, dict):
            self.path_masks = {'mask': to_iterable(mask_url)}
        else:
            self.path_masks = {k: to_iterable(path) for k, path in mask_url.items()}

        self.use_masks = self.path_masks is not None
        self.n_classes = n_classes
        self.filling_strategy = filling_strategy

        super(SegmentationDataset, self).__init__(img_url, shape, keep_size_ratio, recursive_loading,
                                                  extract_image_id_function,
                                                  use_cache)

    def get_class_count(self, save=True, load=True):
        from .utils import get_segmentation_class_count
        return get_segmentation_class_count(self, save=save, load=load)

    def list_files(self, recursive):
        for extension in supportedExtensions:
            prefix = "**/*." if recursive else "*."
            for path in self.path_img:
                self.img_filepath['image'].extend(glob.glob(path + prefix + extension, recursive=recursive))
            if self.use_masks:
                for mask_label, paths in self.path_masks.items():
                    for path in paths:
                        if mask_label not in self.gts:
                            self.gts[mask_label] = []
                        self.gts[mask_label].extend(glob.glob(path + prefix + extension, recursive=recursive))

        self.img_filepath['image'] = np.asarray(self.img_filepath['image'])
        if self.use_masks:
            for mask_label in self.path_masks.keys():
                self.gts[mask_label] = np.asarray(self.gts[mask_label])

        """
        Sorting files
        """
        img_filenames = [path_leaf(path).split('.')[0] for path in self.img_filepath['image']]
        if self.use_masks:
            masks_filenames = {}
            for k, files_list in self.gts.items():
                masks_filenames[k] = [path_leaf(file).split('.')[0] for file in files_list]

            list_lengths = [len(mask_filenames) for mask_filenames in masks_filenames.values()]
            list_lengths.append(len(img_filenames))
            all_equal = all(elem == list_lengths[0] for elem in list_lengths)

            if not all_equal:
                Log.warn("Mismatch between the size of the different input folders (longer %i, smaller %i)" % (max(
                    list_lengths), min(list_lengths)))

            list_common_file = set(img_filenames)
            for k, mask_filenames in masks_filenames.items():
                list_common_file = list_common_file & set(mask_filenames)
            intersection = list(list_common_file)

            if self.filling_strategy == NN_FILL_DOWNSAMPLE and not all_equal:
                Log.warn("Downsampling the dataset to size %i" % min(list_lengths))
                self.img_filepath['image'] = np.asarray(
                    [img for img, filename in zip(self.img_filepath['image'], img_filenames) if
                     filename in intersection])
                for k in self.gts.keys():
                    self.gts[k] = np.asarray(
                        [gt for gt, filename in zip(self.gts[k], masks_filenames[k]) if filename \
                         in intersection])
            elif self.filling_strategy == NN_FILL_UPSAMPLE and not all_equal:
                Log.warn("Upsampling missing labels to fit the dataset's size (%i)" % max(list_lengths))
                for k in self.gts.keys():
                    gt_k = []
                    gt_sorted_filenames = [self.extract_image_id_function(_) for _ in masks_filenames[k]]
                    for img_name in img_filenames:
                        img_name = self.extract_image_id_function(img_name)
                        try:
                            gt_k.append(self.gts[k][gt_sorted_filenames.index(img_name)])
                        except ValueError:
                            gt_k.append(MISSING_DATA_FLAG)
                    self.gts[k] = np.asarray(gt_k)

            if self.filling_strategy == NN_FILL_DOWNSAMPLE or all_equal:
                sort_key_img = np.argsort([self.extract_image_id_function(x) for x in img_filenames])
                self.img_filepath['image'] = self.img_filepath['image'][sort_key_img]

                if self.use_masks:
                    for k in self.gts.keys():
                        sort_key_mask = np.argsort([self.extract_image_id_function(x) for x in masks_filenames[k]])
                        self.gts[k] = self.gts[k][sort_key_mask]

    def load_image(self, item):
        inputs = super(SegmentationDataset, self).load_image(item)
        actual_shape = inputs['image'].shape
        if self.use_masks:
            for k, file_list in self.gts.items():
                filepath = file_list[item]
                if filepath == MISSING_DATA_FLAG and self.filling_strategy == NN_FILL_UPSAMPLE:
                    mask = np.zeros(actual_shape[:-1], dtype=np.uint8)
                else:
                    mask = read_image(filepath, cv2.IMREAD_GRAYSCALE)
                if self.auto_resize:
                    mask = resize(image=mask, shape=self.shape, keep_size_ratio=self.keep_size_ratio,
                                  flag=cv2.INTER_NEAREST_EXACT)
                inputs[k] = mask

        return inputs

    def get_mask(self, item):
        filepath = self.gts[item]
        mask = read_image(filepath, cv2.IMREAD_GRAYSCALE)
        if self.auto_resize:
            mask = resize(image=mask, shape=self.shape, keep_size_ratio=self.keep_size_ratio, flag=cv2.INTER_NEAREST)
        if self.composer:
            mask = self.composer(mask=mask)
        if self.return_indices:
            return mask, item
        else:
            return mask
