import copy
import os

import cv2
import numpy as np
from torch import randperm, default_generator
from torch._utils import _accumulate

from nntools.tracker import Log


def get_classification_class_count(dataset):
    gts = dataset.gts
    unique, count = np.unique(gts, return_counts=True)
    return count


def get_segmentation_class_count(dataset, save=True, load=True):
    shape = dataset.shape
    path = dataset.path_masks[0]
    filepath = os.path.join(path, 'classes_count.npy')

    if os.path.isfile(filepath) and load:
        return np.load(filepath)
    list_masks = dataset.gts

    classes_counts = np.zeros(1024, dtype=int)  # Arbitrary large number (nb classes unknown at this point)

    for f in list_masks:
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, dsize=shape, interpolation=cv2.INTER_NEAREST)
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
