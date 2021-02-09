import inspect
from functools import partial


def to_iterable(param, iterable_type=list):
    if not isinstance(param, iterable_type):
        param = iterable_type([param])
    return param


def partial_fill_kwargs(func, list_args):
    kwargs = {}
    for p in inspect.signature(func).parameters.values():
        if p.name in list_args:
            kwargs[p.name] = list_args[p.name]
    return partial(func, **kwargs)


def call_with_filtered_kwargs(func, list_args):
    kwargs = {}
    for p in inspect.signature(func).parameters.values():
        if p.name in list_args:
            kwargs[p.name] = list_args[p.name]
    return func(**kwargs)
