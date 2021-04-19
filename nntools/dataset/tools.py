import numpy as np

from nntools.utils.misc import partial_fill_kwargs


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
            self.ops.append(partial_fill_kwargs(vertical_flip, self.config))

        if check('horizontal_flip'):
            self.ops.append(partial_fill_kwargs(horizontal_flip, self.config))

        if check('random_scale'):
            self.ops.append(partial_fill_kwargs(random_scale, self.config))

        if check('random_rotate'):
            self.ops.append(partial_fill_kwargs(random_rotation, self.config))

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
        self.deactivated = []
    def add(self, *funcs):
        for f in funcs:
            if isinstance(f, DataAugment):
                self.ops.append(f)
            else:
                self.ops.append(partial_fill_kwargs(f, self.config))
        return self

    def deactivate_op(self, index):
        if not isinstance(index, list):
            index = p[index]

        self.deactivated += [index]

    def __call__(self, **kwargs):
        is_mask = 'mask' in kwargs
        is_image = 'image' in kwargs

        for i, op in enumerate(self.ops):
            if i in self.deactivated:
                continue
            out = op(**kwargs)
            if isinstance(out, tuple):
                if is_mask and not is_image:
                    kwargs['mask'] = out[0]
                elif is_mask and is_image:
                    kwargs['image'] = out[0]
                    kwargs['mask'] = out[1]
                else:
                    kwargs['image'] = out[0]
            elif isinstance(out, dict):
                kwargs = out
            else:
                if is_mask:
                    kwargs['mask'] = out
                else:
                    kwargs['image'] = out

        if is_mask and is_image:
            return kwargs['image'], kwargs['mask']
        elif is_mask and not is_image:
            return kwargs['mask']
        else:
            return kwargs['image']

    def __lshift__(self, other):
        return self.add(other)
