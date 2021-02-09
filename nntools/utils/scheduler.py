import torch


def identity(*args):
    return args


def empty(*args):
    pass


def continuous_epoch(epoch, iteration, epoch_size):
    return epoch + iteration / epoch_size


def discrete_epoch(epoch, iteration, epoch_size):
    return epoch


from collections import namedtuple

scheduler = namedtuple('Scheduler', ['func', 'call_on', 'callback'])
SCHEDULERS = {
    'LambdaLR': scheduler(torch.optim.lr_scheduler.LambdaLR, 'on_epoch', empty),
    'MultiplicativeLR': scheduler(torch.optim.lr_scheduler.MultiplicativeLR, 'on_epoch', empty),
    'StepLR': scheduler(torch.optim.lr_scheduler.StepLR, 'on_epoch', empty),
    'MultiStepLR': scheduler(torch.optim.lr_scheduler.MultiStepLR, 'on_epoch', empty),
    'ExponentialLR': scheduler(torch.optim.lr_scheduler.ExponentialLR, 'on_epoch', empty),
    'CosineAnnealingLR': scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, 'on_epoch', continuous_epoch),
    'ReduceLROnPlateau': scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau, 'on_validation', identity),
    'CyclicLR': scheduler(torch.optim.lr_scheduler.CyclicLR, 'on_iteration', identity),
    'OneCycleLR': scheduler(torch.optim.lr_scheduler.OneCycleLR, 'on_iteration', identity),
    'CosineAnnealingWarmRestarts': scheduler(
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 'on_iteration', identity)
}
