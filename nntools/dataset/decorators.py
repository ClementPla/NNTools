import functools


def single_kwarg(func):
    @functools.wraps(func)
    def wrapper_check_args(*args, **kwargs):
        is_mask = 'mask' in kwargs
        if is_mask:
            mask = kwargs.pop('mask')
        img = func(*args, **kwargs)
        if is_mask:
            return img, mask
        else:
            return img

    return wrapper_check_args


def double_kwarg(func):
    @functools.wraps(func)
    def wrapper_check_args(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_check_args
