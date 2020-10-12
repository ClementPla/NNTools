import torch.nn as nn
from .abstract_nn import AbstractNet

def norm_layer(norm):
    return {'batch_norm': nn.BatchNorm2d,
            'sync_batch_norm': nn.SyncBatchNorm,
            'instance_norm': nn.InstanceNorm2d,
            'layer_norm': nn.LayerNorm}[norm]


class ProxyNetwork(AbstractNet):
    def __init__(self, model):
        super(ProxyNetwork, self).__init__()
        self.proxy = model

    def forward(self, *args, **kwargs):
        return self.proxy(*args, **kwargs)


def nnt_format(model):
    return ProxyNetwork(model)
