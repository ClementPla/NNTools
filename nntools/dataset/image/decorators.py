import functools
import inspect


def preprocess(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        list_parameters = inspect.signature(func).parameters.values()
        accepts_mask = any([p.name == 'mask' for p in list_parameters])
        accepts_image = any([p.name == 'image' for p in list_parameters])

        is_mask_in_param = 'mask' in kwargs
        is_image_in_param = 'image' in kwargs

        if not accepts_mask and is_mask_in_param:
            mask = kwargs.pop('mask')
            return func(*args, **kwargs), mask
        if not accepts_image and is_image_in_param:
            image = kwargs.pop('image')
            return image, func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return wrapper









