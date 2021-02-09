import torch


SCHEDULER = {
    'LambdaLR': (torch.optim.lr_scheduler.LambdaLR, 'on_epoch', lambda epoch, iteration, epoch_size: epoch),
    'MultiplicativeLR': (torch.optim.lr_scheduler.MultiplicativeLR, 'on_epoch', lambda epoch, iteration, epoch_size: epoch),
    'StepLR': (torch.optim.lr_scheduler.StepLR, 'on_epoch', lambda epoch, iteration, epoch_size: epoch),
    'MultiStepLR': (torch.optim.lr_scheduler.MultiStepLR, 'on_epoch', lambda epoch, iteration, epoch_size: epoch),
    'ExponentialLR': (torch.optim.lr_scheduler.ExponentialLR, 'on_epoch', lambda epoch, iteration, epoch_size: epoch),
    'CosineAnnealingLR': (torch.optim.lr_scheduler.CosineAnnealingLR, 'on_epoch', lambda epoch, iteration,
                                                                                         epoch_size:
    epoch+iteration/epoch_size),
    'ReduceLROnPlateau': (torch.optim.lr_scheduler.ReduceLROnPlateau, 'on_validation', lambda x:x),
    'CyclicLR': (torch.optim.lr_scheduler.CyclicLR, 'on_iteration', lambda epoch, iteration,
                                                                           epoch_size: None),
    'OneCycleLR': (torch.optim.lr_scheduler.OneCycleLR, 'on_iteration', lambda epoch, iteration,
                                                                           epoch_size: None),
    'CosineAnnealingWarmRestarts': (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, 'on_iteration', lambda epoch, iteration,
                                                                           epoch_size: None)
}