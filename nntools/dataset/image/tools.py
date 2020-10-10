from nntools.utils.misc import convert_function
import numpy as np


class DataAugment:
    def __init__(self, **config):
        self.config = config
        self.ops = []
        self.p = self.config['ratio']

    def auto_init(self):
        if self.config['vertical_flip']:
            from .preprocess import vertical_flip
            self.ops.append(convert_function(vertical_flip, self.config))

        if self.config['horizontal_flip']:
            from .preprocess import horizontal_flip
            self.ops.append(convert_function(horizontal_flip, self.config))

        if self.config['random_rotate']:
            from .preprocess import random_rotation
            self.ops.append(convert_function(random_rotation, self.config))

        if self.config['random_scale']:
            from .preprocess import random_scale
            self.ops.append(convert_function(random_scale, self.config))

        if self.config['random_rotate']:
            from .preprocess import random_rotation
            self.ops.append(convert_function(random_rotation, self.config))

        return self

    def __call__(self, **kwargs):
        is_mask = 'mask' in kwargs
        for op in self.ops:
            if self.p < np.random.uniform():
                if is_mask:
                    img, mask = op(**kwargs)
                    kwargs['img'] = img
                    kwargs['mask'] = mask
                else:
                    img = op(**kwargs)
                    kwargs['img'] = img
        if is_mask:
            return kwargs['img'], kwargs['mask']
        else:
            return kwargs['img']


class Composition:
    def __init__(self, **config):
        self.config = config
        self.ops = []

    def add(self, *funcs):
        for f in funcs:
            if isinstance(f, DataAugment):
                self.ops.append(f)
            else:
                self.ops.append(convert_function(f, self.config))
        return self

    def __call__(self, **kwargs):
        is_mask = 'mask' in kwargs
        for op in self.ops:
            if is_mask:
                img, mask = op(**kwargs)
                kwargs['img'] = img
                kwargs['mask'] = mask
            else:
                img = op(**kwargs)
                kwargs['img'] = img

        if is_mask:
            return kwargs['img'], kwargs['mask']
        else:
            return kwargs['img']

    def __lshift__(self, other):
        return self.add(other)
