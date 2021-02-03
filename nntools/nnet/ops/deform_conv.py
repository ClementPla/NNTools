import torch
from mmdet.ops.dcn.deform_conv import DeformConv, DeformConvFunction
from torch import nn
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


deform_conv = DeformConvFunction.apply


class DeformConvPackWithBias(DeformConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super(DeformConvPackWithBias, self).__init__(in_channels,
                                                 out_channels,
                                                 kernel_size,
                                                 stride=stride,
                                                 padding=padding,
                                                 dilation=dilation,
                                                 groups=groups,
                                                 deformable_groups=deformable_groups,
                                                 bias=False)
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        out = super(DeformConvPackWithBias, self).forward(x, offset)
        if self.use_bias:
            return out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            return out
        

class RestrictedDeformConvPack(DeformConv):
    def __init__(self, *args, **kwargs):
        super(RestrictedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
            offset[:, :] = 0
        else:
            offset[:, self.kernel_size[0] + self.kernel_size[1] // 2] = 0
            offset[:, self.kernel_size[0] * self.kernel_size[1] + self.kernel_size[0] + self.kernel_size[1] // 2] = 0

        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups, self.deformable_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

class RestrictedDeformConvPackWithBias(RestrictedDeformConvPack):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super(RestrictedDeformConvPackWithBias, self).__init__(in_channels,
                                                               out_channels,
                                                               kernel_size,
                                                               stride=stride,
                                                               padding=padding,
                                                               dilation=dilation,
                                                               groups=groups,
                                                               deformable_groups=deformable_groups,
                                                               bias=False)
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))
            self.bias.requires_grad = False

        self.weight.requires_grad = False
        self.save_offset = False

    def forward(self, x):
        out = super(RestrictedDeformConvPackWithBias, self).forward(x)

        if self.use_bias:
            return out + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            return out

