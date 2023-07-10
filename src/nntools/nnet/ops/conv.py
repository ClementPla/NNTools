import torch.nn as nn
import torch.nn.functional as F


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end), mode="reflect")
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=False, bn=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding=0, dilation=dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=bias)

        self.bn = nn.Sequential(bn(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = fixed_padding(x, self.depthwise.kernel_size[0], dilation=self.depthwise.dilation[0])
        return self.bn(self.pointwise(self.depthwise(x)))


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding="auto",
        dilation=1,
        bias=False,
        norm=nn.Identity(),
        activation=nn.ReLU(),
    ):
        super(Conv2d, self).__init__()
        if padding == "auto":
            kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
            pad_total = kernel_size_effective - 1
            padding = pad_total // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if activation is not None:
            self.bn = nn.Sequential(norm, activation)
        else:
            self.bn = norm

    def forward(self, x):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2, **kwargs):
        super(ResidualBlock, self).__init__()
        convs = []
        for i in range(t):
            if not i:
                convs.append(Conv2d(in_channels, out_channels, **kwargs))
            else:
                convs.append((Conv2d(out_channels, out_channels, **kwargs)))
        self.convs = nn.Sequential(*convs)
        if in_channels != out_channels:
            self.conv_1x1 = Conv2d(in_channels, out_channels, kernel_size=1, activation=nn.Identity())
        else:
            self.conv_1x1 = nn.Identity()

    def forward(self, x):
        conv_x = self.convs(x)
        x = self.conv_1x1(x)
        return conv_x + x
