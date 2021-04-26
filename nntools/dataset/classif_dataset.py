import glob
import os

import numpy as np
import torch
import tqdm

from .image_dataset import ImageDataset, supportedExtensions


class ClassificationDataset(ImageDataset):
    def __init__(self, img_url,
                 shape=None,
                 keep_size_ratio=False,
                 recursive_loading=True,
                 map_class=None,
                 label_present=True,
                 label_per_folder=True,
                 csv_filepath=None,
                 file_column='image',
                 gt_column='level',
                 sort_function=None, use_cache=False):
        self.map_class = map_class
        self.label_present = label_present
        self.label_per_folder = label_per_folder
        self.csv_filepath = csv_filepath
        self.file_column = file_column
        self.gt_column = gt_column

        super(ClassificationDataset, self).__init__(img_url, shape, keep_size_ratio, recursive_loading,
                                                    sort_function, use_cache)

    def list_files(self, recursive):
        for extension in supportedExtensions:
            prefix = "**/*." if recursive else "*."
            for path in self.path_img:
                self.img_filepath['image'].extend(glob.glob(path + prefix + extension, recursive=recursive))

        self.img_filepath['image'] = np.asarray(self.img_filepath['image'])

        if self.label_present:
            if self.label_per_folder:
                for f in self.img_filepath['image']:
                    self.gts.append(os.path.basename(os.path.dirname(f)))
            if self.csv_filepath:
                import pandas
                csv = pandas.read_csv(self.csv_filepath)
                img_names = [os.path.basename(p) for p in self.img_filepath['image']]
                argsort = np.argsort(img_names)
                self.img_filepath['image'] = self.img_filepath['image'][argsort]
                csv_names = np.asarray(csv[self.file_column])
                argsort = np.argsort(csv_names)
                csv_gts = np.asarray(csv[self.gt_column])
                self.gts = csv_gts[argsort]

            unique_labels = np.unique(self.gts)
            self.n_classes = len(unique_labels)
            self.gts = np.asarray(self.gts)

            if self.map_class is None:
                self.map_class = {unique_labels[i]: i for i in np.arange(len(unique_labels))}
            for k, v in self.map_class.items():
                self.gts[self.gts == k] = v
        self.gts = self.gts.astype(int)

    def cache(self):
        self.use_cache = False
        self.sharred_imgs = self.init_cache()[0]
        print('Caching dataset...')
        for i in tqdm.tqdm(range(1, len(self))):
            img = self.load_array(i)

        self.use_cache = True

    def __getitem__(self, item, torch_cast=True, transpose_img=True, return_indices=True):
        outputs = super(ClassificationDataset, self).__getitem__(item, torch_cast, transpose_img, return_indices)
        if self.label_present:
            outputs['gt'] = torch.tensor(self.gts[item], dtype=torch.long)
        return outputs

    def get_class_count(self):
        from .utils import get_classification_class_count
        return get_classification_class_count(self)
