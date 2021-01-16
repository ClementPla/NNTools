import datetime
from os.path import join, splitext

import torch
from torch import nn

from nntools.utils.io import create_folder, get_most_recent_file


def check_nan(state_dict):
    for k in state_dict:
        if torch.isnan(state_dict[k]).any():
            raise ValueError("Corrupted file")


class AbstractNet(nn.Module):
    def __init__(self, model=None):
        self._today = datetime.datetime.now().date()
        self.savepoint = None
        self.params_group = {}

        super(AbstractNet, self).__init__()
        if model is not None:
            self.network = model

    def save(self, filename='trained_model',
             optimizers=None,
             savepoint=None,
             use_datetime=False, **filestamps):

        if savepoint is not None:
            self.savepoint = savepoint

        if optimizers is not None:
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
        else:
            optimizers = []

        for k, v in filestamps.items():
            filename += '_' + k + '_%f' % v

        if '.' not in filename:
            filename += '.pth'

        path = self.savepoint + '/'

        if use_datetime:
            today = str(self._today)
            path = join(path, today + '/')
        create_folder(path)
        path = join(path, filename)

        save_dict = dict(model_state_dict=self.state_dict())
        for i, optim in enumerate(optimizers):
            save_dict['optim_%i' % i] = optim.state_dict()

        filename, file_extension = splitext(path)
        if not file_extension:
            path = path + '.pth'

        torch.save(save_dict, path)
        return path

    def load(self, path, ignore_nan=False, load_most_recent=False, strict=False, map_location=None):
        if map_location is None:
            map_location = torch.device('cpu')
        if load_most_recent:
            path = get_most_recent_file(path)
        print("Loading model from ", path)

        state_dict = torch.load(path, map_location=map_location)['model_state_dict']
        if not ignore_nan:
            check_nan(state_dict)
        self.load_state_dict(state_dict, strict=strict)

    def get_trainable_parameters(self, lr=None):
        if self.params_group:
            return self.params_group
        else:
            return self.parameters()

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def set_params_group(self, params_group):
        self.params_group = params_group



