import numpy as np

from nntools.utils.misc import convert_function


class DataAugment:
    def __init__(self, **config):
        self.config = config
        self.ops = []
        self.p = self.config['ratio']

    def auto_init(self):
        from nntools.dataset.image_tools import vertical_flip, horizontal_flip, random_scale, random_rotation

        def check(param):
            return param in self.config and self.config[param]

        if check('vertical_flip'):
            self.ops.append(convert_function(vertical_flip, self.config))

        if check('horizontal_flip'):
            self.ops.append(convert_function(horizontal_flip, self.config))

        if check('random_scale'):
            self.ops.append(convert_function(random_scale, self.config))

        if check('random_rotate'):
            self.ops.append(convert_function(random_rotation, self.config))

        return self

    def __call__(self, **kwargs):
        is_mask = 'mask' in kwargs
        for op in self.ops:
            if self.p > np.random.uniform():
                if is_mask:
                    img, mask = op(**kwargs)
                    kwargs['image'] = img
                    kwargs['mask'] = mask
                else:
                    img = op(**kwargs)
                    kwargs['image'] = img
        if is_mask:
            return kwargs['image'], kwargs['mask']
        else:
            return kwargs['image']


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
            out = op(**kwargs)
            if isinstance(out, tuple):
                if is_mask:
                    kwargs['image'] = out[0]
                    kwargs['mask'] = out[1]
                else:
                    kwargs['image'] = out[0]
            elif isinstance(out, dict):
                kwargs = out

        if is_mask:
            return kwargs['image'], kwargs['mask']
        else:
            return kwargs['image']

    def __lshift__(self, other):
        return self.add(other)
