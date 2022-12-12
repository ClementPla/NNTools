import glob
import os

import cv2
import numpy as np

from .abstract_image_dataset import AbstractImageDataset, supportedExtensions


class ClassificationDataset(AbstractImageDataset):
    def __init__(self, img_url,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 map_class=None,
                 label_present=True,
                 label_per_folder=True,
                 csv_filepath=None,
                 file_column='image',
                 gt_column='label',
                 extract_image_id_function=None,
                 use_cache=False,
                 flag=cv2.IMREAD_COLOR):
        self.map_class = map_class
        self.label_present = label_present
        self.label_per_folder = label_per_folder if csv_filepath is None else False
        self.csv_filepath = csv_filepath
        self.file_column = file_column
        if not isinstance(gt_column, list):
            gt_column = [gt_column]
        self.gt_column = gt_column
        super(ClassificationDataset, self).__init__(img_url, shape, keep_size_ratio, recursive_loading,
                                                    extract_image_id_function, use_cache,
                                                    flag=flag)

    def list_files(self, recursive):
        for extension in supportedExtensions:
            prefix = "**/*." if recursive else "*."
            for path in self.path_img:
                self.img_filepath['image'].extend(glob.glob(path + prefix + extension, recursive=recursive))

        self.img_filepath['image'] = np.asarray(self.img_filepath['image'])

        if self.label_present:
            if self.label_per_folder:
                self.gts['label'] = []
                for f in self.img_filepath['image']:
                    self.gts['label'].append(os.path.basename(os.path.dirname(f)))
            if self.csv_filepath:
                import pandas
                csv = pandas.read_csv(self.csv_filepath)
                img_names = [os.path.basename(p) for p in self.img_filepath['image']]
                img_names = [self.extract_image_id_function(_) for _ in img_names]
                argsort = np.argsort(img_names)

                self.img_filepath['image'] = self.img_filepath['image'][argsort]
                csv.sort_values(self.file_column, inplace=True)
                for col in self.gt_column:
                    csv_gts = np.asarray(csv[col])
                    self.gts[col] = csv_gts

            if len(self.gt_column) == 1:
                col_label = self.gt_column[0]
                unique_labels = np.unique(self.gts[col_label])
                self.n_classes = len(unique_labels)
            else:
                unique_labels = self.gt_column
                self.n_classes = len(self.gt_column)
            for k, value in self.gts.items():
                self.gts[k] = np.asarray(value)

            if self.map_class is None:
                self.map_class = {i: unique_labels[i] for i in np.arange(len(unique_labels))}

            if len(self.gt_column) == 1:
                col_label = self.gt_column[0]
                for k, v in self.map_class.items():
                    self.gts[col_label][self.gts[col_label] == v] = k
                self.gts[col_label] = self.gts[col_label].astype(int)

    def get_class_count(self, load=True, save=True):
        # Todo add loading and saving of class counts
        if len(self.gt_column) > 1:
            raise NotImplementedError('Getting the class count for more than one target is not implemented')

        col = self.gt_column[0]
        unique, count = np.unique(self.gts[col], return_counts=True)
        return count

    def remap(self, old_key, new_key):
        try:
            self.gt_column[self.gt_column.index(old_key)] = new_key
        except ValueError:
            pass
        super(ClassificationDataset, self).remap(old_key, new_key)

    def load_image(self, item):
        inputs = super(ClassificationDataset, self).load_image(item)
        for k, v in inputs.items():
            if v.ndim == 2:
                inputs[k] = np.expand_dims(v, 2)
        for k in self.gts.keys():
            inputs[k] = self.gts[k][item]
        return inputs
