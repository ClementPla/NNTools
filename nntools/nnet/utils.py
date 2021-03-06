import torch.nn as nn

from .abstract_nn import AbstractNet


def norm_layer(norm):
    return {'batch_norm': nn.BatchNorm2d,
            'sync_batch_norm': nn.SyncBatchNorm,
            'instance_norm': nn.InstanceNorm2d,
            'layer_norm': nn.LayerNorm}[norm]


def nnt_format(model):
    if not isinstance(model, AbstractNet):
        return AbstractNet(model)
    else:
        return model
