import segmentation_models_pytorch.losses as smp_l
import torch.nn as nn

BINARY_MODE = smp_l.BINARY_MODE
MULTICLASS_MODE = smp_l.MULTICLASS_MODE

SUPPORTED_LOSS = {'CrossEntropy': nn.CrossEntropyLoss,
                  'Dice': smp_l.DiceLoss,
                  'Focal': smp_l.FocalLoss,
                  'Jaccard': smp_l.JaccardLoss,
                  'SoftBinaryCrossEntropy': smp_l.SoftBCEWithLogitsLoss,
                  'SoftCrossEntropy': smp_l.SoftCrossEntropyLoss,
                  'Lovasz': smp_l.LovaszLoss}


def register_loss(key, value):
    global SUPPORTED_LOSS
    SUPPORTED_LOSS[key] = value


class FuseLoss:
    def __init__(self, losses=None, fusion='mean'):
        self.fusion = fusion
        if losses is None:
            losses = []
        if not isinstance(losses, list):
            losses = [losses]
        self.losses = losses

    def __call__(self, *args):
        list_losses = [l(*args) for l in self.losses]
        if self.fusion == 'sum':
            return sum(list_losses)
        if self.fusion == 'mean':
            return sum(list_losses) / len(list_losses)

    # TODO : Add weighting scheme for each loss
    def add(self, loss):
        self.losses.append(loss)

