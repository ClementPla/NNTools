import bisect
import copy
import logging
import os
from typing import List

import numpy as np
import torch
import tqdm
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data import Dataset


def get_segmentation_class_count(dataset, save=False, load=False):
    sample = dataset[0]
    if "mask" not in sample.keys():
        raise NotImplementedError

    path = dataset.path_img[0]
    filepath = os.path.join(path, "classes_count.npy")

    if os.path.isfile(filepath) and load:
        return np.load(filepath)
    classes_counts = np.zeros(1024, dtype=int)  # Arbitrary large number (nb classes unknown at this point)

    for sample in tqdm.tqdm(dataset):
        mask = sample["mask"].numpy()
        if mask.ndim == 3:  # Multilabel -> Multiclass
            arr_tmp = np.argmax(mask, axis=0) + 1
            arr_tmp[mask.max(axis=0) == 0] = 0
            mask = arr_tmp
        u, counts = np.unique(mask, return_counts=True)
        classes_counts[u] += counts

    classes_counts = classes_counts[: np.max(np.nonzero(classes_counts)) + 1]
    if save:
        np.save(filepath, classes_counts)
        logging.warn("Weights stored in " + filepath)
    return classes_counts


def class_weighting(class_count, mode="balanced", ignore_index=-100, eps=1, log_smoothing=1.01, center_mean=0):
    assert mode in ["balanced", "log_prob", "frequency"]
    n_samples = sum([c for i, c in enumerate(class_count) if i != ignore_index])

    if mode == "balanced":
        n_classes = len(np.nonzero(class_count))
        class_weights = n_samples / (n_classes * class_count + eps)
    elif mode == "frequency":
        class_weights = n_samples / class_count

    elif mode == "log_prob":
        p_class = class_count / n_samples
        class_weights = (1 / np.log(log_smoothing + p_class)).astype(np.float32)

    if center_mean:
        class_weights = class_weights - class_weights.mean() + center_mean
    if ignore_index >= 0:
        class_weights[ignore_index] = 0

    return class_weights.astype(np.float32)


def check_dataleaks(*datasets: List[Dataset], raise_exception=True):
    is_okay = True
    cols = {"files": [], "gts": []}
    unfold_datasets = []
    for d in datasets:
        if isinstance(d, ConcatDataset):
            unfold_datasets += d.datasets
        else:
            unfold_datasets.append(d)

    for d in unfold_datasets:
        file_cols, gts_cols = d.columns()
        cols["files"].append(list(file_cols))
        cols["gts"].append(list(gts_cols))

    cols["files"] = list(set.intersection(*map(set, cols["files"])))  # Find intersection of columns
    cols["gts"] = list(set.intersection(*map(set, cols["gts"])))

    for f_col in cols["files"]:
        filenames = []
        for d in unfold_datasets:
            filenames.append(d.filenames[f_col])
        join_file = list(set.intersection(*map(set, filenames)))
        if len(join_file) > 0:
            is_okay = False
            if raise_exception:
                raise ValueError("Found common images between datasets")

    for f_col in cols["gts"]:
        filenames = []
        for d in unfold_datasets:
            filenames.append(d.gt_filenames[f_col])
        join_gt = list(set.intersection(*map(set, filenames)))
        if len(join_gt) > 0:
            is_okay = False
            if raise_exception:
                raise ValueError("Found common groundtruth between datasets")

    if not is_okay:
        return is_okay, join_file, join_gt
    return is_okay


def random_split(dataset, lengths, generator=default_generator):
    if sum(lengths)==1:
        lengths = [int(length*len(dataset)) for length in lengths[:-1]]
        lengths.append(len(dataset)-sum(lengths)) # To prevent rounding error
        
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    datasets = []
    for offset, length in zip(_accumulate(lengths), lengths):
        d = copy.deepcopy(dataset)
        # We need to explicit call the attrs post init callback since deepcopy does not call it
        d.__attrs_post_init__()
        # We also need to explicitely copy the composer
        d.composer = copy.deepcopy(dataset.composer)
        indx = indices[offset - length : offset]
        d.subset(indx)
        
        datasets.append(d)
    return tuple(datasets)


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *args, **kwargs):
        self.post_init = False
        super(ConcatDataset, self).__init__(*args, **kwargs)
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

    def get_class_count(self, load=True, save=True):
        class_count = None
        for d in self.datasets:
            if class_count is None:
                class_count = d.get_class_count(load=load, save=save)
            else:
                class_count += d.get_class_count(load=load, save=save)
        return class_count

    @property
    def composer(self):
        return [d.composer for d in self.datasets]

    def multiply_size(self, factor):
        for d in self.datasets:
            d.multiply_size(factor)

    def init_cache(self):
        for d in self.datasets:
            d.init_cache()

    def __setattr__(self, key, value):
        if key == "post_init":
            super(ConcatDataset, self).__setattr__(key, value)
        if hasattr(self, key) or not self.post_init:
            super(ConcatDataset, self).__setattr__(key, value)
        else:
            for d in self.datasets:
                d.__setattr__(key, value)


def concat_datasets_if_needed(datasets):
    if isinstance(datasets, list):
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]
        return dataset
    else:
        return datasets


def convert_dict_to_plottable(dict_arrays):
    plotted_arrays = {}
    for k, v in dict_arrays.items():
        if isinstance(v, torch.Tensor):
            v = v.numpy()
            if v.ndim == 3:
                v = v.transpose((1, 2, 0))
        plotted_arrays[k] = v
    return plotted_arrays
