import glob
import os
from abc import ABC

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from nntools.nnet import nnt_format
from nntools.tracker import Log, Tracker
from nntools.utils.io import save_yaml
from nntools.utils.misc import partial_fill_kwargs
from nntools.utils.optims import OPTIMS
from nntools.utils.random import set_seed, set_non_torch_seed
from nntools.utils.scheduler import SCHEDULERS
from nntools.utils.torch import DistributedDataParallelWithAttributes as DDP, MultiEpochsDataLoader
from nntools.dataset.utils import concat_datasets_if_needed
from torch.cuda.amp import autocast
from tqdm import tqdm
from nntools.utils import Config
import math


class Manager(ABC):
    def __init__(self, config: Config, run_id: str = None):
        self.config = config
        self.seed = self.config['Manager']['seed']
        set_seed(self.seed)

        self.tracker = Tracker(self.config['Manager']['experiment'], run_id)
        self.tracker.create_client(self.config['Manager']['tracking_uri'], self.config['Manager']['artifact_uri'])
        self.tracker.set_run_folder(os.path.join(self.config['Manager']['save_point'],
                                                 self.config['Manager']['experiment'],
                                                 self.config['Manager']['run']))

        self.model = None
        self.gpu = self.config['Manager']['gpu']
        self.continue_training = True
        self.call_end_function = True
        self.keyboard_exception_raised = False
        if not isinstance(self.gpu, list):
            self.gpu = [self.gpu]

        self.world_size = len(self.gpu)
        self.multi_gpu = self.world_size > 1

        if self.multi_gpu:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

    def convert_batch_norm(self, model: nn.Module) -> nn.Module:
        if self.c['Network']['synchronized_batch_norm'] and self.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def start_run(self, run_id: str = None):
        Log.warn('Initializing tracker')
        tags = {MLFLOW_RUN_NAME: self.config['Manager']['run']}
        if run_id is None and self.tracker.run_id is None:
            self.continue_training = False
            self.tracker.create_run(tags)
        else:
            self.tracker.get_run(run_id)

        if self.continue_training:
            self.tracker.go_to_exp_last_iteration()
        self.tracker.init_default_path()
        self.tracker.set_status('RUNNING')
        Log.warn('Run started (status = RUNNING)')
        self.tracker.run_started = True

    def clean_up(self):
        if self.multi_gpu:
            dist.destroy_process_group()

    def is_main_process(self, rank: int):
        return (rank == 0) or not self.multi_gpu

    def get_gpu_from_rank(self, rank: int):
        return self.gpu[rank]

    def get_model_on_device(self, rank: int) -> nn.Module:
        tqdm.write('Rank %i, gpu %i' % (rank, self.get_gpu_from_rank(rank)))
        torch.cuda.set_device(self.get_gpu_from_rank(rank))
        model = self.get_model()
        model = self.convert_batch_norm(model)
        model = model.cuda(self.get_gpu_from_rank(rank))
        if self.multi_gpu:
            model = DDP(model, device_ids=[self.get_gpu_from_rank(rank)])
        return model

    def get_model(self) -> nn.Module:
        assert self.model is not None, "The model has not been configured, call set_model(model)"
        if self.continue_training:
            self.model.load(self.tracker.network_savepoint, load_most_recent=True, map_location='cpu')

        return self.model

    def set_model(self, model: nn.Module) -> nn.Module:
        model = nnt_format(model)
        self.model = model
        return model

    def set_params_group(self, params_group: dict):
        self.model.set_params_group(params_group)

    def batch_to_device(self, batch: (dict, tuple, list), rank: int ) -> (dict, tuple, list):
        device = self.get_gpu_from_rank(rank)
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [b.cuda(device) if isinstance(b, torch.Tensor) else b for b in batch]
        elif isinstance(batch, dict):
            batch = {k: b.cuda(device) for k, b in batch.items()}
        else:
            batch = batch.cuda(device)
        return batch

    def log_metrics(self, step, **metrics):
        self.tracker.log_metrics(step, **metrics)

    def log_params(self, **params):
        self.tracker.log_params(**params)

    def log_artifacts(self, *paths):
        self.tracker.log_artifacts(*paths)

    def set_tags(self, **tags):
        self.tracker.set_tags(**tags)

    @property
    def c(self):
        return self.config

    @property
    def id(self):
        return self.tracker.run_id

    @property
    def run_started(self):
        return self.tracker.run_started

    @property
    def register_params(self):
        return self.tracker.register_params

    @register_params.setter
    def register_params(self, value):
        self.tracker.register_params = value


class Experiment(Manager):
    def __init__(self, config, run_id=None):
        super(Experiment, self).__init__(config, run_id)

        self.batch_size = self.config['Training']['batch_size'] // self.world_size
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.partial_optimizer = None
        self.partial_lr_scheduler = None

        self.tracked_metric = None
        self.class_weights = None

        self.saved_models = {'best_valid': None, 'last': None}

        self.save_last = True

        self.run_training = True
        self.ctx_train = {}

    def initial_tracking(self):
        self.log_params(**self.config.tracked_params)
        save_yaml(self.config, os.path.join(self.tracker.run_folder, 'config.yaml'))
        self.log_artifacts(self.config.get_path())

    def set_train_dataset(self, dataset):
        dataset = concat_datasets_if_needed(dataset)
        self.train_dataset = dataset

    def set_valid_dataset(self, dataset):
        dataset = concat_datasets_if_needed(dataset)
        self.validation_dataset = dataset

    def set_test_dataset(self, dataset):
        self.test_dataset = dataset

    def create_optimizer(self, **config):
        solver = config['solver']
        func = OPTIMS[solver]
        return partial_fill_kwargs(func, config['params_solver'])

    def set_optimizer(self, optimizers=None, **config):
        """
        :return: A partial function of an optimizers. Partial passed arguments are hyper parameters
        """
        if optimizers is not None:
            self.partial_optimizer = optimizers
        else:
            self.partial_optimizer = self.create_optimizer(**config)

    def set_scheduler(self, **config):
        scheduler_name = config['scheduler']
        scheduler = SCHEDULERS[scheduler_name]
        self.ctx_train['scheduler_opt'] = scheduler
        self.partial_lr_scheduler = partial_fill_kwargs(scheduler.func, config['params_scheduler'])

    def get_dataloader(self, dataset, shuffle=True, batch_size=None, drop_last=False, persistent_workers=True, rank=0):
        num_workers = self.config['Manager']['num_workers']
        exp_dataloader = self.config['Manager'].get('experimental_dataloader', False)
        c_shuffle = self.config['Dataset'].get('shuffle', True)
        shuffle = shuffle & c_shuffle

        if batch_size is None:
            batch_size = self.batch_size
        if self.multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle,
                                                                      num_replicas=self.world_size, rank=rank)
        else:
            sampler = None

        if not exp_dataloader:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pin_memory=True, shuffle=shuffle if sampler is None else False,
                                                     sampler=sampler,
                                                     worker_init_fn=set_non_torch_seed, drop_last=drop_last,
                                                     persistent_workers=persistent_workers if num_workers else False)
        else:
            dataloader = MultiEpochsDataLoader(dataset, batch_size=batch_size,
                                                     num_workers=num_workers,
                                                     pin_memory=True, shuffle=shuffle if sampler is None else False,
                                                     sampler=sampler,
                                                     worker_init_fn=set_non_torch_seed, drop_last=drop_last,
                                                     persistent_workers=persistent_workers if num_workers else False)

        return dataloader, sampler

    def save_model(self, model, filename, **kwargs):
        tqdm.write('Saving model')
        save = model.save(savepoint=self.tracker.network_savepoint, filename=filename, **kwargs)

        if 'best_valid' in filename:
            self.saved_models['best_valid'] = save
        else:
            self.saved_models['last'] = save

        if self.config['Manager']['max_saved_model']:
            files = glob.glob(self.tracker.network_savepoint + "/best_valid_*.pth")
            files.sort(key=os.path.getmtime)
            for f in files[:-self.config['Manager']['max_saved_model']]:
                os.remove(f)

    def _start_process(self, rank: int = 0):
        tqdm.write('Initializing process %i' % rank)
        if self.multi_gpu:
            dist.init_process_group(backend=self.config['Manager']['dist_backend'], rank=rank,
                                    world_size=self.world_size)
        model = self.get_model_on_device(rank)
        if self.run_training:
            try:
                self.train(model, rank)
            except KeyboardInterrupt:
                self.keyboard_exception_raised = True
            finally:
                self.tracker.set_status('FAILED')

            if self.keyboard_exception_raised:
                if self.is_main_process(rank):
                    Log.warn("Killed Process. The last model will be registered at %s" % self.saved_models)
                    self.tracker.set_status('KILLED')

        if self.is_main_process(rank) and (self.run_training or self.save_last):
            self.save_model(model, 'last')
            self.register_trained_model()

        if self.multi_gpu:
            dist.barrier()

        if self.call_end_function:
            with autocast(enabled=self.config['Manager']['amp']):
                self.end(model, rank)

        self.clean_up()

    def start(self, run_id=None):
        if not self.run_started:
            self.start_run(run_id)

        if self.run_training:
            assert self.partial_optimizer is not None, "Missing optimizer for training"
            assert self.train_dataset is not None, "Missing dataset"
            if self.validation_dataset is None:
                Log.warn("Missing validation set, default behaviour is to save the model once per epoch")

            if self.partial_lr_scheduler is None:
                Log.warn("Missing learning rate scheduler, default behaviour is to keep the learning rate constant")

        self.keyboard_exception_raised = False

        if self.register_params:
            self.initial_tracking()

        if self.multi_gpu:
            mp.spawn(self._start_process, nprocs=self.world_size, join=True)

        else:
            self._start_process()

        if not self.keyboard_exception_raised:
            self.tracker.set_status(status='FINISHED')

    def register_trained_model(self):
        if self.saved_models['best_valid']:
            self.log_artifacts(self.saved_models['best_valid'])
        if self.saved_models['last']:
            self.log_artifacts(self.saved_models['last'])

    def lr_scheduler_step(self, lr_scheduler, epoch, iteration, size_epoch, validation_metrics=None):
        if lr_scheduler is None:
            pass
        elif self.ctx_train['scheduler_opt'].call_on == 'on_validation':
            lr_scheduler.step(validation_metrics)
        else:
            lr_scheduler.step(self.ctx_train['scheduler_opt'].callback(epoch, iteration, size_epoch))

    def eval(self, register_params=False, run_id=None):
        self.register_params = register_params
        self.run_training = False
        self.start(run_id=run_id)

    def train(self, model, rank):
        pass

    def validate(self, model, valid_loader, iteration, rank=0, loss_function=None):
        pass

    def end(self, model, rank):
        pass

    def in_epoch(self, *args, **kwargs):
        pass

    def main_training_loop(self, rank=0):
        total_epoch = self.config['Training'].get('epochs', -1)
        max_iterations = self.config['Training'].get('iterations', -1)
        assert (total_epoch > 0) or (max_iterations > 0), "You must define a number of training iterations or a number " \
                                                          "of epochs"
        if max_iterations > 0:
            total_epoch = math.ceil(max_iterations / math.ceil(len(self.train_dataset)/self.batch_size))

        for e in range(total_epoch):
            if self.is_main_process(rank):
                tqdm.write('** Epoch %i/%i **' % (e, total_epoch))
                self.log_metrics(e, progress=100 * e / total_epoch)
            self.in_epoch(epoch=e, rank=rank)
