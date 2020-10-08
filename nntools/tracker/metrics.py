import numpy as np
import torch
import torch.nn.functional as F


class Metrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.metrics_list = []

    def __call__(self, pred, gt):
        output = {}
        for m in self.metrics_list:
            name = m.__name__
            output[name] = m(pred, gt, self.n_classes)
        return output

    def append(self, metrics):
        if isinstance(metrics, list):
            self.metrics_list += metrics
        else:
            self.metrics_list.append(metrics)


def confusion_matrix(pred, gt, num_classes=-1, *args):
    """
    :param pred: Tensor of any shape containing hard predictions (integer)
    :param gt: Tensor of same shape as pred
    :param num_classes:
    :return:
    """
    pred_one_hot = F.one_hot(pred.flatten(), num_classes)
    gt_one_hot = F.one_hot(gt.flatten(), num_classes)
    return torch.matmul(pred_one_hot.t(), gt_one_hot)


def mIoU(pred, gt, num_classes=-1, epsilon=1e-5, *args):
    confMat = confusion_matrix(pred, gt, num_classes)
    intersection = torch.diag(confMat)
    union = confMat.sum(0) + confMat.sum(1) - intersection
    return torch.mean(intersection / (union + epsilon))


def mIoU_cm(confMat, epsilon=1e-5):
    intersection = torch.diag(confMat)
    union = confMat.sum(0) + confMat.sum(1) - intersection
    return torch.mean(intersection / (union + epsilon))


def filter_index_cm(confMat, index):
    n_classes = confMat.shape[0]
    classes = np.arange(n_classes)
    confMat = confMat[classes != index]
    confMat = confMat[:, classes != index]
    return confMat


def report_cm(confMat, epsilon=1e-7):
    all = confMat.sum()
    TP = torch.diag(confMat)
    P = confMat.sum(0)
    N = all - P
    PredP = confMat.sum(1)
    FP = PredP - TP
    FN = P - TP
    TN = all - TP - FP - FN
    sensitivity = TP / (P + epsilon)
    specificity = TN / (N + epsilon)
    precision = TP / (TP + FP + epsilon)
    return {'sensitivity': sensitivity.mean(), 'specificity': specificity.mean(), 'precision': precision.mean()}
