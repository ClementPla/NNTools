import segmentation_models_pytorch.losses as smp_l
import torch.nn as nn
from nntools import MULTICLASS_MODE, MULTILABEL_MODE, BINARY_MODE

SUPPORTED_LOSS = {'CrossEntropy': nn.CrossEntropyLoss,
                  'Dice': smp_l.DiceLoss,
                  'Focal': smp_l.FocalLoss,
                  'Jaccard': smp_l.JaccardLoss,
                  'SoftBinaryCrossEntropy': smp_l.SoftBCEWithLogitsLoss,
                  'SoftCrossEntropy': smp_l.SoftCrossEntropyLoss,
                  'Lovasz': smp_l.LovaszLoss,
                  'NLL': nn.NLLLoss}


def register_loss(key, value):
    global SUPPORTED_LOSS
    SUPPORTED_LOSS[key] = value


def auto_register(func):
    global SUPPORTED_LOSS
    SUPPORTED_LOSS[func.__name__] = func
    return func


class FuseLoss:
    def __init__(self, losses=None, fusion='mean', mode=MULTICLASS_MODE):

        self.mode = mode
        self.fusion = fusion
        if losses is None:
            losses = []

        if not isinstance(losses, list):
            losses = [losses]
        self.losses = losses

    def __call__(self, *y_pred, y_true):
        list_losses = []
        for l in self.losses:
            list_losses.append(l(*y_pred, y_true))
        if self.fusion == 'sum':
            return sum(list_losses)
        if self.fusion == 'mean':
            return sum(list_losses) / len(list_losses)

    def add(self, loss):
        self.losses.append(loss)
