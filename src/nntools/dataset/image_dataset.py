import logging
from timeit import default_timer as timer
from typing import Literal

import numpy as np
from attrs import define

from nntools.dataset.abstract_image_dataset import AbstractImageDataset
from nntools.utils.const import NNOpt
from nntools.utils.io import list_files_in_folder, path_leaf
from nntools.utils.misc import to_iterable


@define
class MultiImageDataset(AbstractImageDataset):
    filling_strategy: Literal[NNOpt.FILL_DOWNSAMPLE, NNOpt.FILL_UPSAMPLE] = (
        NNOpt.FILL_DOWNSAMPLE
    )

    def list_files(self, recursive):
        if not isinstance(self.img_root, dict):
            img_root = {"image": to_iterable(self.img_root)}
        else:
            img_root = {}
            for k, v in self.img_root.items():
                self.on_disk_keys.add(k)
                img_root[k] = to_iterable(v)

        self.img_filepath = {k: [] for k in img_root.keys()}
        start = timer()
        for root_label, paths in img_root.items():
            for path in paths:
                filepaths = list_files_in_folder(path, recursive=recursive)
                self.img_filepath[root_label].extend(filepaths)
        end = timer()
        logging.debug(f"Listing files took {end - start}")
        if len(self.img_filepath.keys()) == 1:
            # Always store as ndarray so downstream fancy indexing (subset,
            # filename, ...) works uniformly regardless of the number of keys.
            for k in self.img_filepath:
                self.img_filepath[k] = np.asarray(self.img_filepath[k])
        if len(self.img_filepath.keys()) > 1:
            imgs_ids = {}
            start = timer()
            for k, filepaths in self.img_filepath.items():
                self.img_filepath[k] = np.asarray(filepaths)
                imgs_ids[k] = np.asarray(
                    [
                        self.extract_image_id_function(path_leaf(path))
                        for path in self.img_filepath[k]
                    ]
                )
                argsort_ids = np.argsort(imgs_ids[k])
                imgs_ids[k] = imgs_ids[k][argsort_ids]
                self.img_filepath[k] = self.img_filepath[k][argsort_ids]
            end = timer()
            logging.debug(f"Sorting files took {end - start}")

            start = timer()
            list_lengths = [len(img_ids) for img_ids in imgs_ids.values()]
            all_equal = all(elem == list_lengths[0] for elem in list_lengths)
            if not all_equal:
                logging.warning(
                    "Mismatch between the size of the different input folders (longer %i, smaller %i)"
                    % (max(list_lengths), min(list_lengths))
                )
                logging.debug(
                    f"List lengths: {list(zip(list(imgs_ids.keys()), list_lengths))}"
                )

            list_common_file = set.intersection(*map(set, list(imgs_ids.values())))
            intersection_ids = np.asarray(list(list_common_file))
            logging.debug(
                f"Number of files in intersection dataset: {len(intersection_ids)}"
            )
            end = timer()
            logging.debug(f"Finding common files took {end - start}")
            if self.filling_strategy == NNOpt.FILL_DOWNSAMPLE or all_equal:
                start = timer()
                # We only keep the intersection of the files
                if not all_equal:
                    logging.warning(
                        "Downsampling the dataset to size %i" % min(list_lengths)
                    )
                for k, ids in imgs_ids.items():
                    self.img_filepath[k] = self.img_filepath[k][
                        np.isin(ids, intersection_ids)
                    ]

                end = timer()
                logging.debug(f"Downsampling number of files took {end - start}")

            elif self.filling_strategy == NNOpt.FILL_UPSAMPLE and not all_equal:
                start = timer()
                # Union of every id across all folders, sorted (np.unique sorts).
                union_ids = np.unique(
                    np.concatenate([np.asarray(ids) for ids in imgs_ids.values()])
                )

                for k, ids in imgs_ids.items():
                    # Which union positions this key actually provides. Both
                    # union_ids and this key's filepaths are sorted by id, so
                    # the boolean mask aligns them directly.
                    present = np.isin(union_ids, ids)
                    filepaths = np.empty(len(union_ids), dtype=object)
                    filepaths[present] = self.img_filepath[k]
                    filepaths[~present] = NNOpt.MISSING_DATA_FLAG
                    self.img_filepath[k] = filepaths

                end = timer()
                logging.debug(f"Upsampling number of files took {end - start}")

    def __len__(self):
        if self.filling_strategy == NNOpt.FILL_DOWNSAMPLE:
            return min([len(filepaths) for filepaths in self.img_filepath.values()])
        elif self.filling_strategy == NNOpt.FILL_UPSAMPLE:
            return max([len(filepaths) for filepaths in self.img_filepath.values()])


@define(slots=False)
class ImageDataset(MultiImageDataset):
    pass
