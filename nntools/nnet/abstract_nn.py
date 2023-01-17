import datetime
import os
from typing import Dict, Sequence, Union

import torch
from nntools.utils.io import create_folder, get_most_recent_file
from torch import nn
from torchmetrics import MetricCollection, Metric


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
        self._metrics = MetricCollection([])

        if model is not None:
            self.network = model

    def get_savepath(self, filename, savepoint, use_datetime=False, **filestamps):
        if savepoint is not None:
            self.savepoint = savepoint

        for k, v in filestamps.items():
            filename += '_' + k + '_%f' % v

        if '.' not in filename:
            filename += '.pth'

        path = self.savepoint

        if use_datetime:
            today = str(self._today)
            path = os.path.join(path, today)

        create_folder(path)
        path = os.path.join(path, filename)

        filename, file_extension = os.path.splitext(path)
        if not file_extension:
            path = path + '.pth'

        return path

    def save(self, filename='trained_model',
             optimizers=None,
             savepoint=None,
             use_datetime=False, **filestamps):

        if optimizers is not None:
            if not isinstance(optimizers, list):
                optimizers = [optimizers]
        else:
            optimizers = []
        save_dict = dict(model_state_dict=self.state_dict())
        for i, optim in enumerate(optimizers):
            save_dict['optim_%i' % i] = optim.state_dict()

        path = self.get_savepath(filename, savepoint, use_datetime, **filestamps)
        torch.save(save_dict, path)
        return path

    # def save_scripted(self, filename='model_scripted', savepoint=None, use_datetime=False):
    #     model_scripted = torch.jit.script(self)
    #     path = self.get_savepath(filename, savepoint=savepoint, use_datetime=use_datetime)
    #     model_scripted.save(path)

    def load(self, path, ignore_nan=False, load_most_recent=False, strict=False, map_location=None, filtername=None,
             allow_size_mismatch=True,
             verbose=True):
        if map_location is None:
            map_location = torch.device('cpu')
        if load_most_recent:
            path = get_most_recent_file(path, filtername)
        if verbose:
            print("Loading model from ", path)

        state_dict = torch.load(path, map_location=map_location)['model_state_dict']
        if allow_size_mismatch and not strict:
            current_model = self.state_dict()
            new_state_dict = {
                k: state_dict.get(k, None) if (k in state_dict and (state_dict[k].size() == v.size())) else v for k, v
                in current_model.items()}
            state_dict = new_state_dict

        if not ignore_nan:
            check_nan(state_dict)
        return self.load_state_dict(state_dict, strict=strict)

    def get_trainable_parameters(self):
        if self.params_group:
            return self.params_group
        else:
            return self.parameters()

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def set_params_group(self, params_group):
        self.params_group = params_group

    def add_metric(self, metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]], *additional_metrics: Metric):
        self._metrics.add_metrics(metrics, *additional_metrics)
