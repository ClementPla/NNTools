import functools
import inspect

import cv2
import numpy as np


def nntools_wrapper(func):
    @functools.wraps(func)
    def wrapper(**kwargs):
        expected_parameters = inspect.signature(func).parameters.values()
        arguments = {}
        for p in expected_parameters:
            if p.name in kwargs:
                arguments[p.name] = kwargs.pop(p.name)
            elif p.default is not p.empty:
                arguments[p.name] = p.default
        output = func(**arguments)
        output.update(kwargs)
        return output

    return wrapper


def resize(image, keep_size_ratio=True, shape=(512, 512), flag=cv2.INTER_LINEAR):
    if isinstance(shape, int):
        shape = (shape, shape)
    else:
        shape = tuple(shape)
    if keep_size_ratio:
        dims = image.shape[:2]
        f = np.min([d2/d1 for d1, d2 in zip(dims, shape)])
        shape = (int(f*dims[0]), int(f*dims[1]))

    image = cv2.resize(image, dsize=shape, interpolation=flag)
    return image
