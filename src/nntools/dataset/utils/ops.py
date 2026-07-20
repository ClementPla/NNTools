import copy

import numpy as np
from torch import default_generator, randperm

from nntools.dataset.viewer import Viewer


def random_split(dataset, lengths, generator=default_generator):
    if sum(lengths) == 1:
        lengths = [int(length * len(dataset)) for length in lengths[:-1]]
        lengths.append(len(dataset) - sum(lengths))  # To prevent rounding error

    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(len(dataset), generator=generator).tolist()
    split_idxs = [
        indices[offset - length : offset]
        for offset, length in zip(np.cumsum(lengths), lengths)
    ]

    datasets = split(dataset, split_idxs)
    return tuple(datasets)


def split(dataset, indices):
    datasets = []
    for split, indx in enumerate(indices):
        d = copy.deepcopy(dataset)
        d.img_filepath = copy.deepcopy(dataset.img_filepath)
        d.gts = copy.deepcopy(dataset.gts)
        d.composer = copy.deepcopy(dataset.composer)
        d.ignore_keys = copy.deepcopy(dataset.ignore_keys)
        d.viewer = Viewer(d)
        d.subset(indx)
        d.id = dataset.id + f"_split_{split}"
        d.create_cache()
        datasets.append(d)
    return tuple(datasets)
