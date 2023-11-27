import inspect
import numbers
from functools import partial
from typing import Any, Sequence


def to_iterable(param: Any, iterable_type: Sequence[Any]=list):
    if isinstance(param, dict):
        return param
    if not isinstance(param, iterable_type):
        param = iterable_type([param])
    return param


def partial_fill_kwargs(func, list_args):
    kwargs = {}
    for p in inspect.signature(func).parameters.values():
        if p.name in list_args:
            kwargs[p.name] = list_args[p.name]
    return partial(func, **kwargs)


def call_with_filtered_kwargs(func, dict_args):
    kwargs = {}
    for p in inspect.signature(func).parameters.values():
        if p.name in dict_args:
            kwargs[p.name] = dict_args[p.name]
    return func(**kwargs)


def identity(x):
    return x


def tensor2num(x):
    if isinstance(x, numbers.Number):
        return x
    if x.dim() == 0:
        return x.item()
    else:
        return x.detach().cpu().numpy()
