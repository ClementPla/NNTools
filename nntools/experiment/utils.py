import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed):
    torch.manual_seed(seed)
    cudnn.deterministic = True
    set_non_torch_seed(seed)


def set_non_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
