def to_iterable(param, iterable_type=list):
    if not isinstance(param, iterable_type):
        param = iterable_type([param])
    return param


import torch.distributed as dist


def reduce_tensor(tensor, world_size, mode='avg'):
    rt = tensor.clone
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    if mode == 'avg':
        return rt / world_size
    elif mode == 'sum':
        return rt
    else:
        raise NotImplementedError
