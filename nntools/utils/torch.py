import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

class DistributedDataParallelWithAttributes(DDP):
    """
    Allow nn.DataParallel to call model's attributes.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def reduce_tensor(tensor, world_size, mode='avg'):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'avg':
        return rt / world_size
    elif mode == 'sum':
        return rt
    else:
        raise NotImplementedError




class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
