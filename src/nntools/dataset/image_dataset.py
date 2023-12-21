import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
from attrs import define

from nntools import MISSING_DATA_FLAG, NN_FILL_DOWNSAMPLE, NN_FILL_UPSAMPLE
from nntools.dataset.abstract_image_dataset import AbstractImageDataset, supportedExtensions
from nntools.utils.misc import to_iterable


@define
class MultiImageDataset(AbstractImageDataset):
    filling_strategy: Literal["NN_FILL_DOWNSAMPLE", "NN_FILL_UPSAMPLE"] = NN_FILL_DOWNSAMPLE

    def list_files(self, recursive):
        if not isinstance(self.img_root, dict):
            img_root = {"image": to_iterable(self.img_root)}
        else:
            img_root = {}
            for k, v in self.img_root.items():
                img_root[k] = to_iterable(v)

        self.img_filepath = {k: [] for k in img_root.keys()}
        start = timer()
        prefix = "**/*" if recursive else "*"
        for root_label, paths in img_root.items():
            for path in paths:
                filepaths = [p.resolve() for p in Path(path).glob(prefix) if p.suffix in supportedExtensions]
                self.img_filepath[root_label].extend(filepaths)
        end = timer()
        logging.debug(f'Listing files took {end - start}')
        if len(self.img_filepath.keys()) > 1:
            imgs_ids = {}
            start = timer()
            for k, filepaths in self.img_filepath.items():
                self.img_filepath[k] = np.asarray(filepaths)
                imgs_ids[k] = np.asarray([self.extract_image_id_function(path.stem) for path in self.img_filepath[k]])
                argsort_ids = np.argsort(imgs_ids[k])
                imgs_ids[k] = imgs_ids[k][argsort_ids]
                self.img_filepath[k] = self.img_filepath[k][argsort_ids]
            end = timer()
            logging.debug(f'Sorting files took {end - start}')
            
            start = timer()
            list_lengths = [len(img_ids) for img_ids in imgs_ids.values()]
            all_equal = all(elem == list_lengths[0] for elem in list_lengths)
            if not all_equal:
                logging.warning(
                    "Mismatch between the size of the different input folders (longer %i, smaller %i)"
                    % (max(list_lengths), min(list_lengths))
                )
                logging.debug(f"List lengths: {list(zip(list(imgs_ids.keys()), list_lengths))}")

            list_common_file = set.intersection(*map(set, list(imgs_ids.values())))
            intersection_ids = np.asarray(list(list_common_file))
            logging.debug(f"Number of files in intersection dataset: {len(intersection_ids)}")
            end = timer()
            logging.debug(f'Finding common files took {end - start}')
            if self.filling_strategy == NN_FILL_DOWNSAMPLE or all_equal:
                start = timer()
                # We only keep the intersection of the files
                if not all_equal:
                    logging.warning("Downsampling the dataset to size %i" % min(list_lengths))
                for k, ids in imgs_ids.items():
                    self.img_filepath[k] = self.img_filepath[k][np.isin(ids, intersection_ids)]
                
                end = timer()
                logging.debug(f'Downsampling number of files took {end - start}')

            elif self.filling_strategy == NN_FILL_UPSAMPLE and not all_equal:
                start = timer()
                union_ids = np.asarray(set.union(*map(set, list(imgs_ids.values)))).sort()

                for k, v in imgs_ids.items():
                    temps_ids = np.isin(v, union_ids)
                    img_filepath = np.zeros(len(union_ids), dtype=v.dtype)
                    img_filepath[temps_ids] = self.img_filepath[k]
                    img_filepath[~temps_ids] = MISSING_DATA_FLAG
                    self.img_filepath[k] = img_filepath
                
                end = timer()
                logging.debug(f'Upsampling number of files took {end - start}')

    def __len__(self):
        if self.filling_strategy == NN_FILL_DOWNSAMPLE:
            return min([len(filepaths) for filepaths in self.img_filepath.values()])
        elif self.filling_strategy == NN_FILL_UPSAMPLE:
            return max([len(filepaths) for filepaths in self.img_filepath.values()])


@define(slots=False)
class ImageDataset(MultiImageDataset):
    pass
