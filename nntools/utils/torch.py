from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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