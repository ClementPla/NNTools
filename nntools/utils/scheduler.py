import torch


def pass_args(*args):
    return args


def block_args(*args):
    pass


from collections import namedtuple

Scheduler = namedtuple('Scheduler', ['func', 'callback'])

SCHEDULERS = {
    'LambdaLR': Scheduler(torch.optim.lr_scheduler.LambdaLR, block_args),
    'MultiplicativeLR': Scheduler(torch.optim.lr_scheduler.MultiplicativeLR, block_args),
    'StepLR': Scheduler(torch.optim.lr_scheduler.StepLR, block_args),
    'MultiStepLR': Scheduler(torch.optim.lr_scheduler.MultiStepLR, block_args),
    'ExponentialLR': Scheduler(torch.optim.lr_scheduler.ExponentialLR, block_args),
    'CosineAnnealingLR': Scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, block_args),
    'ReduceLROnPlateau': Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau, pass_args),
    'CyclicLR': Scheduler(torch.optim.lr_scheduler.CyclicLR, block_args),
    'OneCycleLR': Scheduler(torch.optim.lr_scheduler.OneCycleLR, block_args),
    'CosineAnnealingWarmRestarts': Scheduler(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, block_args)
}
