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

Scheduler = namedtuple('Scheduler', ['func', 'call_on', 'callback'])

SCHEDULERS = {
    'LambdaLR': Scheduler(torch.optim.lr_scheduler.LambdaLR, 'on_epoch', empty),
    'MultiplicativeLR': Scheduler(torch.optim.lr_scheduler.MultiplicativeLR, 'on_epoch', empty),
    'StepLR': Scheduler(torch.optim.lr_scheduler.StepLR, 'on_epoch', empty),
    'MultiStepLR': Scheduler(torch.optim.lr_scheduler.MultiStepLR, 'on_epoch', empty),
    'ExponentialLR': Scheduler(torch.optim.lr_scheduler.ExponentialLR, 'on_epoch', empty),
    'CosineAnnealingLR': Scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, 'on_epoch', continuous_epoch),
    'ReduceLROnPlateau': Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau, 'on_validation', identity),
    'CyclicLR': Scheduler(torch.optim.lr_scheduler.CyclicLR, 'on_iteration', identity),
    'OneCycleLR': Scheduler(torch.optim.lr_scheduler.OneCycleLR, 'on_iteration', identity),
    'CosineAnnealingWarmRestarts': Scheduler(
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 'on_iteration', identity)
}
