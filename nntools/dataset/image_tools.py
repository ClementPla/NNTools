import functools
import inspect

import cv2
import numpy as np


def nntools_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        list_parameters = inspect.signature(func).parameters.values()
        params_name = ([p.name for p in list_parameters])

        arguments = {k: kwargs.pop(k) for k in params_name if k in kwargs}
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
        h, w = image.shape[:2]
        max_size = np.max((h, w))
        scale = max(shape) / max_size
        shape = (int(scale * w), int(scale * h))

    image = cv2.resize(image, dsize=shape, interpolation=flag)
    return image
