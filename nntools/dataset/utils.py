import bisect
import copy
import os

import numpy as np
import torch
import tqdm
from torch import randperm, default_generator
from torch._utils import _accumulate

from nntools.tracker import Log


def get_segmentation_class_count(dataset, save=True, load=True):
    sample = dataset[0]
    if 'mask' not in sample.keys():
        raise NotImplementedError

    path = dataset.path_img[0]
    filepath = os.path.join(path, 'classes_count.npy')

    if os.path.isfile(filepath) and load:
        return np.load(filepath)
    classes_counts = np.zeros(1024, dtype=int)  # Arbitrary large number (nb classes unknown at this point)

    for sample in tqdm.tqdm(dataset):
        mask = sample['mask'].numpy()
        if mask.ndim == 3:  # Multilabel -> Multiclass
            arr_tmp = np.argmax(mask, axis=0) + 1
            arr_tmp[mask.max(axis=0) == 0] = 0
            mask = arr_tmp
        u, counts = np.unique(mask, return_counts=True)
        classes_counts[u] += counts

    classes_counts = classes_counts[:np.max(np.nonzero(classes_counts)) + 1]
    if save:
        np.save(filepath, classes_counts)
        Log.warn('Weights stored in ' + filepath)
    return classes_counts


def class_weighting(class_count, mode='balanced', ignore_index=-100, eps=1, log_smoothing=1.01, center_mean=0):
    assert mode in ['balanced', 'log_prob']
    n_samples = sum([c for i, c in enumerate(class_count) if i != ignore_index])

    if mode == 'balanced':
        n_classes = len(np.nonzero(class_count))
        class_weights = n_samples / (n_classes * class_count + eps)

    elif mode == 'log_prob':
        p_class = class_count / n_samples
        class_weights = (1 / np.log(log_smoothing + p_class)).astype(np.float32)

    if center_mean:
        class_weights = class_weights - class_weights.mean() + center_mean
    if ignore_index >= 0:
        class_weights[ignore_index] = 0

    return class_weights.astype(np.float32)


def random_split(dataset, lengths, generator=default_generator):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    datasets = []
    for offset, length in zip(_accumulate(lengths), lengths):
        d = copy.deepcopy(dataset)
        indx = indices[offset - length: offset]
        d.subset(indx)
        datasets.append(d)
    return tuple(datasets)


class ConcatDataset(torch.utils.data.ConcatDataset):

    def __init__(self, *args, **kwargs):
        self.post_init = False
        super(ConcatDataset, self).__init__(*args, **kwargs)
        self.post_init = True

    def plot(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        self.datasets[dataset_idx].plot(sample_idx)

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

    def cache(self):
        for d in self.datasets:
            d.cache()

    def __setattr__(self, key, value):
        if key == 'post_init':
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
