from typing import List, Optional, Union

import numpy as np
import pandas
from attrs import define, field

from nntools.dataset.abstract_image_dataset import AbstractImageDataset
from nntools.utils.io import list_files_in_folder, path_folder_leaf, path_leaf
from nntools.utils.misc import to_iterable


@define
class ClassificationDataset(AbstractImageDataset):
    map_class: Optional[Union[dict[int, str], dict[str, dict[int, str]]]] = None
    label_dataframe: Optional[pandas.DataFrame] = None
    label_filepath: Optional[str] = None
    label_per_folder: bool = field()

    @label_per_folder.default
    def _label_per_folder_default(self):
        return self.label_filepath is None and self.label_dataframe is None

    file_column: str = "image"
    gt_column: Union[str, List[str]] = field(default="label", converter=to_iterable)

    def list_files(self, recursive):
        for path in self.img_root:
            filepaths = list_files_in_folder(path, recursive=recursive)
            self.img_filepath["image"].extend(filepaths)

        self.img_filepath["image"] = np.asarray(self.img_filepath["image"])

        if self.label_per_folder:
            # Get the name of the containing folder
            self.gts["label"] = [
                path_folder_leaf(p) for p in self.img_filepath["image"]
            ]
        elif self.label_dataframe is not None:
            self.match_df_with_images(self.label_dataframe)

        elif self.label_filepath:
            if self.label_filepath.endswith(".xls"):
                df_labels = pandas.read_excel(self.label_filepath)
            else:
                df_labels = pandas.read_csv(self.label_filepath)
            self.match_df_with_images(df_labels)

        else:
            return

        for k, value in self.gts.items():
            self.gts[k] = np.asarray(value)

        self.auto_remap_to_integer_label()

    def auto_remap_to_integer_label(self):
        if self.map_class is None:
            self.map_class = dict()
            for k in self.gt_column:
                unique_labels = np.unique(self.gts[k])
                self.map_class[k] = {i: label for i, label in enumerate(unique_labels)}

        for k in self.gts.keys():
            for new_label, old_label in self.map_class[k].items():
                self.gts[k][self.gts[k] == old_label] = new_label
            self.gts[k] = self.gts[k].astype(int)

    @property
    def n_classes(self):
        if isinstance(self.gt_column, str):
            col_label = self.gt_column
        else:
            col_label = self.gt_column[0]
        unique_labels = np.unique(self.gts[col_label])
        return len(unique_labels)

    def match_df_with_images(self, df: pandas.DataFrame):
        img_names = np.array(
            [
                self.extract_image_id_function(path_leaf(p))
                for p in self.img_filepath["image"]
            ]
        )
        df_img = np.array(df[self.file_column].map(lambda x: path_leaf(x)))

        if len(df_img) != len(img_names):
            img_argsort = np.argsort(img_names)
            df_argsort = np.argsort(df_img)

            common = np.intersect1d(img_names, df_img)
            common = np.sort(common)
            pos_img = np.searchsorted(img_names[img_argsort], common)
            pos_df = np.searchsorted(df_img[df_argsort], common)

            img_indices = img_argsort[pos_img]
            df_indices = df_argsort[pos_df]

            img_names = img_names[img_indices]
            self.img_filepath["image"] = self.img_filepath["image"][img_indices]
            df = df.iloc[df_indices].copy()

        argsort = np.argsort(img_names)
        self.img_filepath["image"] = self.img_filepath["image"][argsort]
        df.sort_values(self.file_column, inplace=True)
        for col in self.gt_column:
            df_gts = np.asarray(df[col])
            self.gts[col] = df_gts

    def get_class_count(self, load=True, save=True):
        # Todo Add loading and saving of class counts
        # Todo Add support for more than one target class
        if len(self.gt_column) > 1:
            raise NotImplementedError(
                "Getting the class count for more than one target is not implemented"
            )

        col = self.gt_column[0]
        unique, count = np.unique(self.gts[col], return_counts=True)
        return count

    def remap(self, old_key, new_key):
        self.gt_column[self.gt_column.index(old_key)] = new_key
        super(ClassificationDataset, self).remap(old_key, new_key)

    def read_from_disk(self, item):
        inputs = super(ClassificationDataset, self).read_from_disk(item)
        for k, v in inputs.items():
            if v.ndim == 2:
                inputs[k] = np.expand_dims(v, 2)
        for k in self.gts.keys():
            inputs[k] = self.gts[k][item]
        return inputs

    def filter_classes(self, classes: List[int]):
        if isinstance(classes, int):
            classes = [classes]

        for k in self.gts.keys():
            all_classes = np.unique(self.gts[k])
            classes_to_keep = np.setdiff1d(all_classes, classes)
            kept_indices = np.isin(self.gts[k], classes_to_keep)
            self.img_filepath["image"] = self.img_filepath["image"][kept_indices]
            for k in self.gts.keys():
                self.gts[k] = self.gts[k][kept_indices]

        if self.use_cache:
            self.cache.d = self
