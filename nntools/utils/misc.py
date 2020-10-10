import inspect
from functools import partial


def to_iterable(param, iterable_type=list):
    if not isinstance(param, iterable_type):
        param = iterable_type([param])
    return param


def convert_function(func, list_args):
    kwargs = {}
    for p in inspect.signature(func).parameters.values():
        if p.name in list_args:
            kwargs[p.name] = list_args[p.name]
    return partial(func, **kwargs)
