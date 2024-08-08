import bisect
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from nntools.dataset.abstract_image_dataset import AbstractImageDataset

from nntools.dataset.viewer import Viewer


def concat_datasets_if_needed(datasets):
    if isinstance(datasets, list):
        if len(datasets) == 0:
            return None
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]
        return dataset
    else:
        return datasets


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *args, **kwargs):
        self.post_init = False
        self.datasets: list[AbstractImageDataset]
        super().__init__(*args, **kwargs)
        self.viewer = Viewer(self)
        self.post_init = True

    def plot(self, idx, **kwargs):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        self.datasets[dataset_idx].plot(sample_idx, **kwargs)

    def get_mosaic(
        self,
        n_items: int = 9,
        shuffle: bool = False,
        indexes: Optional[List[int]] = None,
        resolution: Tuple[int, int] = (512, 512),
        show: bool = False,
        fig_size: int = 1,
        save: Optional[bool] = None,
        add_labels: bool = False,
        n_row: Optional[int] = None,
        n_col: Optional[int] = None,
        n_classes: Optional[int] = None,
    ):
        return self.viewer.get_mosaic(
            n_items, shuffle, indexes, resolution, show, fig_size, save, add_labels, n_row, n_col, n_classes
        )

    def get_class_count(self, load=True, save=True):
        class_count = None
        for d in self.datasets:
            if class_count is None:
                class_count = d.get_class_count(load=load, save=save)
            else:
                class_count += d.get_class_count(load=load, save=save)
        return class_count

    def __getitem__(self, idx, return_indices=False, return_tag=False):
        return super().__getitem__(idx)

    @property
    def composer(self):
        return [d.composer for d in self.datasets]

    @composer.setter
    def composer(self, other):
        for d in self.datasets:
            d.composer = other

    def multiply_size(self, factor):
        for d in self.datasets:
            d.multiply_size(factor)

    def init_cache(self):
        for d in self.datasets:
            d.init_cache()

    def __setattr__(self, key, value):
        if key == "post_init":
            super().__setattr__(key, value)
        elif not self.post_init:
            super().__setattr__(key, value)
        elif self.post_init:
            for d in self.datasets:
                d.__setattr__(key, value)

    def __getattr__(self, item):
        if item not in [
            "init_cache",
            "composer",
            "get_class_count",
            "multiply_size",
            "datasets",
            "get_mosaic",
            "plot",
            "viewer",
        ]:
            return [getattr(d, item) for d in self.datasets]
