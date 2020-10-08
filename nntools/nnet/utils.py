import torch.nn as nn


def norm_layer(norm):
    return {'batch_norm': nn.BatchNorm2d,
            'sync_batch_norm': nn.SyncBatchNorm,
            'instance_norm': nn.InstanceNorm2d,
            'layer_norm': nn.LayerNorm}[norm]
