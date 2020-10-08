import numpy as np

from .preprocess import ImageTransform


class DataAugment:
    def __init__(self, config):
        self.config = config
        self.ops = []
        self.p = self.config['ratio']

    def auto_init(self):
        ImageTransform.cval = self.config['cval']
        if self.config['horizontal_flip']:
            self.ops.append(ImageTransform.horizontal_flip)

        if self.config['random_scale']:
            ImageTransform.scale_factor = self.config['scale_range']
            self.ops.append(ImageTransform.random_scale)

        if self.config['random_rotate']:
            ImageTransform.rotation_angle = self.config['rotation_range']
            self.ops.append(ImageTransform.random_rotation)
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
    def __init__(self, config):
        self.config = config
        if 'crop_size' in config:
            ImageTransform.crop_size = self.config['crop_size']
        self.ops = []

    def add(self, *args):
        for a in args:
            self.ops.append(a)
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
