import logging
from typing import Callable, Literal, Optional, Union

import cv2
import numpy as np
from attrs import define, field

from nntools.dataset.abstract_image_dataset import AbstractImageDataset
from nntools.dataset.functional.geometry import resize
from nntools.dataset.utils.balance import get_segmentation_class_count
from nntools.utils.const import NNOpt
from nntools.utils.io import list_files_in_folder, path_leaf, read_image
from nntools.utils.misc import to_iterable
from pathlib import Path


def extract_filename_without_extension(filename: str | Path):
    if isinstance(filename, Path):
        filename = str(filename)
    return filename.split(".")[0]


def mask_path_converter(mask_path):
    if mask_path is None or mask_path == "":
        return None
    elif not isinstance(mask_path, dict):
        return {"mask": to_iterable(mask_path)}
    else:
        return {k: to_iterable(path) for k, path in mask_path.items()}


@define
class SegmentationDataset(AbstractImageDataset):
    extract_image_id_function: Optional[Callable] = field(
        default=extract_filename_without_extension
    )
    mask_root: Optional[Union[str, dict[str, str]]] = field(
        default=None, converter=mask_path_converter
    )
    use_masks: bool = field()

    @use_masks.default
    def _use_masks_default(self):
        return self.mask_root is not None

    filling_strategy: Literal[NNOpt.FILL_DOWNSAMPLE, NNOpt.FILL_UPSAMPLE] = field(
        default=NNOpt.FILL_DOWNSAMPLE, converter=NNOpt
    )

    binarize_mask: bool = field(default=False)
    n_classes: Optional[int] = field(default=None)

    def get_class_count(self, save=False, load=False):
        return get_segmentation_class_count(self, save=save, load=load)

    def list_files(self, recursive):
        if self.extract_image_id_function is None:
            self.extract_image_id_function = extract_filename_without_extension

        for path in self.img_root:
            filepaths = list_files_in_folder(path, recursive=recursive)
            self.img_filepath["image"].extend(filepaths)
        if self.use_masks:
            for mask_label, paths in self.mask_root.items():
                self.on_disk_keys.add(mask_label)
                for path in paths:
                    if mask_label not in self.gts:
                        self.gts[mask_label] = []
                    filepaths = list_files_in_folder(path, recursive=recursive)
                    self.gts[mask_label].extend(filepaths)

        if self.use_masks:
            gts_ids = {}
            for mask_key in self.mask_root.keys():
                self.gts[mask_key] = np.asarray(self.gts[mask_key])
                gts_ids[mask_key] = [
                    self.extract_image_id_function(
                        path.relative_to(self.mask_root[mask_key][0]).with_suffix("")
                    )
                    for path in self.gts[mask_key]
                ]
                argsort_ids = np.argsort(gts_ids[mask_key])
                gts_ids[mask_key] = np.asarray(gts_ids[mask_key])[argsort_ids]
                self.gts[mask_key] = self.gts[mask_key][argsort_ids]

        self.img_filepath["image"] = np.asarray(self.img_filepath["image"])

        imgs_ids = []
        for path in self.img_filepath["image"]:
            for root in self.img_root:
                try:
                    filename = path.relative_to(root)
                except ValueError:
                    continue
                imgs_ids.append(
                    self.extract_image_id_function(filename.with_suffix(""))
                )

        img_ids = np.asarray(imgs_ids)
        argsort_ids = np.argsort(img_ids)
        img_ids = img_ids[argsort_ids]
        self.img_filepath["image"] = self.img_filepath["image"][argsort_ids]

        if self.use_masks:
            list_lengths = [len(mask_ids) for mask_ids in gts_ids.values()] + [
                len(img_ids)
            ]
            all_equal = all(elem == list_lengths[0] for elem in list_lengths)

            if not all_equal:
                logging.warning(
                    "Mismatch between the size of the different input folders (longer %i, smaller %i)"
                    % (max(list_lengths), min(list_lengths))
                )
                logging.debug(
                    f"List lengths: {list(zip([*list(gts_ids.keys()), 'image'], list_lengths))}"
                )

            list_common_file = set(img_ids)
            for mask_ids in gts_ids.values():
                list_common_file = list_common_file & set(mask_ids)
            intersection_ids = np.asarray(list(list_common_file))
            logging.debug(
                f"Number of files in intersection dataset: {len(intersection_ids)}"
            )
            if len(intersection_ids) == 0:
                logging.warning("No common files between the different folders")
                for k in self.gts.keys():
                    logging.debug(f"List of files in {k}: {gts_ids[k]}")
                logging.debug(f"List of files in image: {img_ids}")

            if self.filling_strategy == NNOpt.FILL_DOWNSAMPLE or all_equal:
                # We only keep the intersection of the files
                if not all_equal:
                    logging.warning(
                        "Downsampling the dataset to size %i" % min(list_lengths)
                    )

                self.img_filepath["image"] = self.img_filepath["image"][
                    np.isin(img_ids, intersection_ids)
                ]

                for k in self.gts.keys():
                    self.gts[k] = self.gts[k][np.isin(gts_ids[k], intersection_ids)]

            elif self.filling_strategy == NNOpt.FILL_UPSAMPLE and not all_equal:
                if len(img_ids) < max(list_lengths):
                    raise ValueError(
                        "Upsampling is not possible if the dataset is smaller than the biggest folder"
                    )

                logging.warning(
                    "Upsampling missing labels to fit the dataset's size (%i)"
                    % len(img_ids)
                )
                for k, values in self.gts.items():
                    temps_ids = np.isin(img_ids, gts_ids[k])
                    gts_k = np.zeros(len(img_ids), dtype=values.dtype)
                    gts_k[temps_ids] = values
                    gts_k[~temps_ids] = NNOpt.MISSING_DATA_FLAG.value
                    self.gts[k] = gts_k

    def read_from_disk(self, item: int):
        inputs = super(SegmentationDataset, self).read_from_disk(item)
        actual_shape = inputs["image"].shape
        if self.use_masks:
            for k, file_list in self.gts.items():
                inputs[k] = self.load_mask(item, k, actual_shape[:-1])
        return inputs

    def get_mask(self, item: int, key: str = "mask"):
        mask = self.load_mask(
            item,
            key,
        )
        if self.composer:
            mask = self.composer(mask=mask)
        if self.return_indices:
            return mask, item
        else:
            return mask

    def load_mask(
        self,
        item: int,
        key: str = "mask",
        expected_shape: Optional[tuple[int, int]] = None,
    ):
        if expected_shape is None:
            expected_shape = self.shape
        filepath = self.gts[key][item]
        if filepath == NNOpt.MISSING_DATA_FLAG.value:
            mask = np.zeros(expected_shape, dtype=np.uint8)
        else:
            mask = read_image(filepath, cv2.IMREAD_GRAYSCALE)

        mask = self.resize_and_pad(mask, interpolation=cv2.INTER_NEAREST_EXACT)

        if self.binarize_mask:
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        return mask


@define
class SegmentationDatasetWithColorMask(SegmentationDataset):
    color_interpretation: dict[tuple[int, int, int], int] = field(default=None)
    method: Literal["OR", "INDEXING", "WHERE", "VECTORIALIZE"] = field(
        default="INDEXING"
    )
    # This was implemented for test purposes, to compare speed

    def load_mask(
        self,
        item: int,
        key: str = "mask",
        expected_shape: Optional[tuple[int, int]] = None,
    ):
        filepath = self.gts[key][item]
        if filepath == NNOpt.MISSING_DATA_FLAG.value:
            mask = np.zeros(expected_shape, dtype=np.uint8)
        else:
            mask = read_image(filepath, cv2.IMREAD_COLOR_RGB)

        if mask.ndim == 3:
            mask = self.map_color_to_class(mask, self.color_interpretation)

        mask = self.resize_and_pad(mask, interpolation=cv2.INTER_NEAREST_EXACT)

        if self.binarize_mask:
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        return mask

    def map_color_to_class(self, mask: np.ndarray, color: tuple[int, int, int]) -> int:
        # mask as dimensions HxWxC

        if self.color_interpretation is None:
            raise ValueError("Color interpretation is not defined")

        colors = np.asarray(list(self.color_interpretation.keys()))
        colors_id = np.asarray(list(self.color_interpretation.values()))

        # Print the unique tuples of colors present in mask
        result = np.zeros(mask.shape[:2], dtype=np.uint8)
        if self.method == "VECTORIALIZE":
            R, C, D = np.where((mask == colors[:, None, None, :]).all(3))
            result[C, D] = colors_id[R]
            return result

        for color, class_id in zip(colors, colors_id):
            _ma = (mask == color).all(axis=2)

            if self.method == "OR":
                result = result * (~_ma) + class_id * (_ma)
            elif self.method == "INDEXING":
                result[_ma] = class_id
            elif self.method == "WHERE":
                result = np.where(_ma, class_id, result)

        return result
