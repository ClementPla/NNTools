import torch


OPTIMS = {'Adam': torch.optim.Adam,
          'AdamW': torch.optim.AdamW,
          'Adadelta': torch.optim.Adadelta,
          'Adagrad': torch.optim.Adagrad,
          'SparseAdam': torch.optim.SparseAdam,
          'Adamax': torch.optim.Adamax,
          'ASGD': torch.optim.ASGD,
          'LBFGS': torch.optim.LBFGS,
          'RMSprop': torch.optim.RMSprop,
          'Rprop': torch.optim.Rprop,
          'SGD': torch.optim.SGD}