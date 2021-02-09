import torch


def identity(*args):
    return args


def empty(*args):
    pass


def continuous_epoch(epoch, iteration, epoch_size):
    return epoch + iteration / epoch_size


def discrete_epoch(epoch, iteration, epoch_size):
    return epoch


SCHEDULER = {
    'LambdaLR': (torch.optim.lr_scheduler.LambdaLR, 'on_epoch', empty),
    'MultiplicativeLR': (torch.optim.lr_scheduler.MultiplicativeLR, 'on_epoch', empty),
    'StepLR': (torch.optim.lr_scheduler.StepLR, 'on_epoch', empty),
    'MultiStepLR': (torch.optim.lr_scheduler.MultiStepLR, 'on_epoch', empty),
    'ExponentialLR': (torch.optim.lr_scheduler.ExponentialLR, 'on_epoch', empty),
    'CosineAnnealingLR': (torch.optim.lr_scheduler.CosineAnnealingLR, 'on_epoch', continuous_epoch),
    'ReduceLROnPlateau': (torch.optim.lr_scheduler.ReduceLROnPlateau, 'on_validation', identity),
    'CyclicLR': (torch.optim.lr_scheduler.CyclicLR, 'on_iteration', identity),
    'OneCycleLR': (torch.optim.lr_scheduler.OneCycleLR, 'on_iteration', identity),
    'CosineAnnealingWarmRestarts': (
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 'on_iteration', identity)
}
