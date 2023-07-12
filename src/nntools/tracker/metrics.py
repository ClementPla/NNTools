import numpy as np
import torch
import torch.nn.functional as F


def confusion_matrix(pred, gt, num_classes=-1, multilabel=False, *args):
    """
    :param pred: Tensor of any of the following shape containing hard predictions (integer or bool): NxHxW,
    Nx(W*W) (multi
    class problem), NxCxHxW or NxCx(H*W) (multi-label problem).
    :param gt: Tensor of same shape as pred
    :param num_classes:
    :param multilabel If true, will consider the second dimension as being the class dimension. In this case,
    gt and pred can only contains binary values.
    :return: A matrix of size CxC for multiclass or Cx2x2 for multilabel.
    """
    if multilabel:
        b, c = pred.shape[:2]
        conf_matrix = torch.zeros((c, 2, 2)).to(pred.device)
        for i in range(c):
            conf_matrix[i] = confusion_matrix(pred[:, i].long(), gt[:, i].long(), num_classes=2, multilabel=False)
        return conf_matrix
    if num_classes == -1:
        num_classes = max(pred.max(), gt.max()) + 1
    pred_one_hot = F.one_hot(pred.flatten(), num_classes)
    gt_one_hot = F.one_hot(gt.flatten(), num_classes)
    return torch.matmul(pred_one_hot.t().float(), gt_one_hot.float()).long()


def mIoU(pred, gt, num_classes=-1, epsilon=1e-7, multilabel=False, *args):
    confMat = confusion_matrix(pred, gt, num_classes, multilabel=multilabel)
    return mIoU_cm(confMat, epsilon=epsilon)


def mIoU_cm(confMat, epsilon=1e-7):
    if confMat.ndim == 3:
        return sum([mIoU_cm(c, epsilon=epsilon) for c in confMat]) / confMat.shape[0]
    else:
        intersection = torch.diag(confMat)
        union = confMat.sum(0) + confMat.sum(1) - intersection
        return torch.mean(intersection / (union + epsilon)).item()


def filter_index_cm(confMat, index):
    n_classes = confMat.shape[0]
    classes = np.arange(n_classes)
    confMat = confMat[classes != index]
    if confMat.ndim == 2:
        confMat = confMat[:, classes != index]
    return confMat


def micro_score(confMat, epsilon=1e-7):
    TP, TN, P, N, FP, FN = extract_confMat_values(confMat)
    sensitivity_m = TP.sum() / (P.sum() + epsilon)
    specificity_m = TN.sum() / (N.sum() + epsilon)
    precision_m = TP.sum() / (TP.sum() + FP.sum() + epsilon)
    accuracy_m = (TP.sum() + TN.sum()) / (P.sum() + N.sum() + epsilon)
    f1_m = (2 * TP.sum()) / (2 * TP.sum() + FP.sum() + FN.sum() + epsilon)
    return {
        "sensitivity_micro": sensitivity_m.item(),
        "specificity_micro": specificity_m.item(),
        "precision_micro": precision_m.item(),
        "accuracy_micro": accuracy_m.item(),
        "f1_micro": f1_m.item(),
    }


def macro_score(confMat, epsilon=1e-7):
    TP, TN, P, N, FP, FN = extract_confMat_values(confMat)
    sensitivity = TP / (P + epsilon)
    specificity = TN / (N + epsilon)
    precision = TP / (TP + FP + epsilon)
    accuracy = (TP + TN) / (P + N + epsilon)
    f1 = 2 * TP / (2 * TP + FP + FN + epsilon)
    return {
        "sensitivity_macro": sensitivity.mean().item(),
        "specificity_macro": specificity.mean().item(),
        "precision_macro": precision.mean().item(),
        "accuracy_macro": accuracy.mean().item(),
        "f1_macro": f1.mean().item(),
    }


def extract_confMat_values(confMat):
    if confMat.ndim == 3:
        TP = confMat[:, 1, 1]
        TN = confMat[:, 0, 0]
        FP = confMat[:, 0, 1]
        FN = confMat[:, 1, 0]
        P = FN + TP
        N = FP + TN
    elif confMat.ndim == 2:
        all = confMat.sum()
        TP = torch.diag(confMat)
        P = confMat.sum(0)
        N = all - P
        PredP = confMat.sum(1)
        FP = PredP - TP
        FN = P - TP
        TN = all - TP - FP - FN
    else:
        ValueError(
            "Confusion matrix can only have 2 (multiclass) or 3 (multilabels) dimension. Got shape ", confMat.shape
        )
    return TP, TN, P, N, FP, FN


def report_cm(confMat, epsilon=1e-7, macro=True, micro=False):
    score = {}
    if macro:
        score = macro_score(confMat, epsilon)
    if micro:
        score.update(micro_score(confMat, epsilon))
    return score
