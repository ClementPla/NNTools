import segmentation_models_pytorch.losses as smp_l
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import BINARY_MODE, MULTICLASS_MODE

BINARY_MODE = BINARY_MODE
MULTICLASS_MODE = MULTICLASS_MODE

SUPPORTED_LOSS = {'CrossEntropy': nn.CrossEntropyLoss,
                  'Dice': smp_l.DiceLoss,
                  'Focal': smp_l.FocalLoss,
                  'Jaccard': smp_l.JaccardLoss,
                  'SoftBinaryCrossEntropy': smp_l.SoftBCEWithLogitsLoss,
                  'SoftCrossEntropy': smp_l.SoftCrossEntropyLoss,
                  'Lovasz': smp_l.LovaszLoss}


class FuseLoss:
    def __init__(self, losses=None, fusion='mean'):
        self.fusion = fusion
        if losses is None:
            losses = []
        if not isinstance(losses, list):
            losses = [losses]
        self.losses = losses

    def __call__(self, x, y):
        list_losses = [l(x, y) for l in self.losses]
        if self.fusion == 'sum':
            return sum(list_losses)
        if self.fusion == 'mean':
            return sum(list_losses) / len(list_losses)

    # TODO : Add weighting scheme for each loss
    def add(self, loss):
        self.losses.append(loss)


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits, true):
        """Computes the Sørensen–Dice loss.
            Note that PyTorch optimizers minimize a loss. In this
            case, we would like to maximize the dice loss so we
            return the negated dice loss.
            Args:
                true: a tensor of shape [B, 1, H, W].
                logits: a tensor of shape [B, C, H, W]. Corresponds to
                    the raw output or logits of the model.
                eps: added to the denominator for numerical stability.
            Returns:
                dice_loss: the Sørensen–Dice loss.
            """
        num_classes = logits.shape[1]
        # TODO: take in account the ignore index
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).contiguous()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).contiguous()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(probas.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()

        return 1 - dice_loss
