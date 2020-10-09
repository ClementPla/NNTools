import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..abstract_nn import AbstractNet
from ..ops import Conv2d


class DeepLabv3Plus(AbstractNet):
    def __init__(self, config):
        super(DeepLabv3Plus, self).__init__()
        self.config = config
        self.output_stride = config['output_stride']
        self.decoder = Decoder(256, self.config['n_classes'])
        self._init_weight()
        self.encoder = Encoder(self.config)

    def forward(self, x):
        b, c, h, w = x.shape
        x, high_res_feat = self.encoder(x, True)
        pred = self.decoder(x, high_res_feat)
        return F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_1x_lr_params(self):

        modules = [self.encoder.encoder]

        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for name, p in m[1].named_parameters():
                    if p.requires_grad:
                        yield p

    def get_10x_lr_params(self):

        modules = [self.encoder.aspp, self.decoder]

        for i in range(len(modules)):
            for m in modules[i].named_modules():
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p

    def get_trainable_parameters(self, lr=0.001):
        return [{'params': self.get_1x_lr_params(), 'lr': lr}, {'params': self.get_10x_lr_params(), 'lr': lr * 10}]


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Decoder, self).__init__()
        bn = nn.BatchNorm2d
        self.feat_red = Conv2d(256, 48, 1, norm=bn(48))
        self.classifier = nn.Sequential(Conv2d(48 + in_channels, 256, kernel_size=3),
                                        nn.Dropout(0.5),
                                        Conv2d(256, 256, kernel_size=3, norm=bn(256)),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(256, n_classes, 1))

    def forward(self, x, low_res_feat):
        decoder_f = self.feat_red(low_res_feat)
        x = F.interpolate(x, size=decoder_f.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, decoder_f), dim=1)
        return self.classifier(x)


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}

from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.utils import load_state_dict_from_url


def _segm_resnet(name, backbone_name, num_classes, pretrained_backbone=True):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out', 'layer1': 'features'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def _load_model(arch_type, backbone, pretrained, progress, num_classes, **kwargs):
    model = _segm_resnet(arch_type, backbone, num_classes, **kwargs)
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict, strict=False)
    return model


def deeplabv3_resnet101(pretrained=True, progress=True,
                        num_classes=21, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _load_model('deeplabv3', 'resnet101', pretrained, progress, num_classes, **kwargs)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        net = deeplabv3_resnet101(pretrained=self.config['pretrained'])
        encoder = net.backbone
        aspp = net.classifier[0]
        if self.config['output_stride'] == 16:
            layer = getattr(encoder, 'layer3')
            for name, module in layer.named_modules():
                if name == '0.downsample.0' or name == '0.conv1':
                    module.stride = (2, 2)

                if not name.startswith('0'):
                    if isinstance(module, nn.Conv2d):
                        if module.dilation != (1, 1):
                            module.dilation = tuple(np.asarray(module.dilation) // 2)
                            module.padding = module.dilation

            layer = getattr(encoder, 'layer4')
            for name, module in layer.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.dilation != (1, 1):
                        module.dilation = tuple(np.asarray(module.dilation) // 2)
                        module.padding = module.dilation

            for name, module in aspp.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.dilation != (1, 1):
                        module.dilation = tuple(np.asarray(module.dilation) // 2)
                        module.padding = tuple(np.asarray(module.padding) // 2)

        self.encoder = encoder
        self.aspp = aspp

    def forward(self, x, features=True):
        result = self.encoder(x)
        x = self.aspp(result['out'])
        if features:
            return x, result['features']
        else:
            return x
