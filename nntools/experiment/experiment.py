import glob
import math
import os
from abc import ABC
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from nntools.dataset.utils import concat_datasets_if_needed
from nntools.nnet import nnt_format
from nntools.tracker import Log, Tracker
from nntools.utils import Config
from nntools.utils.io import save_yaml
from nntools.utils.misc import partial_fill_kwargs, tensor2num
from nntools.utils.optims import OPTIMS
from nntools.utils.plotting import create_mosaic, plt_cmap
from nntools.utils.random import set_seed, set_non_torch_seed
from nntools.utils.scheduler import SCHEDULERS
from nntools.utils.torch import DistributedDataParallelWithAttributes as DDP, MultiEpochsDataLoader
from torch.cuda.amp import GradScaler
from torch.profiler import ProfilerActivity
from tqdm.auto import tqdm
from optuna.integration.pytorch_distributed import TorchDistributedTrial


class Manager(ABC):
    def __init__(self, config: Config, run_id: str = None):
        self.config = config
        self.seed = self.config['Manager']['seed']
        set_seed(self.seed)

        self.tracker = Tracker(self.config['Manager']['experiment'], run_id)
        self.tracker.create_client(
            self.config['Manager']['tracking_uri'], self.config['Manager']['artifact_uri'])
        self.tracker.set_run_folder(os.path.join(self.config['Manager']['save_point'],
                                                 self.config['Manager']['experiment'],
                                                 self.config['Manager']['run']))

        self.model = None
        self.gpu = self.config['Manager']['gpu']
        self.continue_training = True
        self.call_end_function = True
        self.keyboard_exception_raised = False
        self.save_jit_model = False

        self.run_profiling = False

        if not isinstance(self.gpu, list):
            self.gpu = [self.gpu]

        self.world_size = len(self.gpu)
        self.multi_gpu = self.world_size > 1

        if self.multi_gpu:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

        self.ddp_config = self.config.get('DDP', dict())
        self.ctx = Context(multi_gpu=self.multi_gpu, additional_dataloader={})

    def start_run(self, run_id: str = None):
        Log.warn('Initializing tracker')
        tags = {MLFLOW_RUN_NAME: self.config['Manager']['run']}
        if run_id is None and self.tracker.run_id is None:
            self.continue_training = False
            self.tracker.create_run(tags)
        else:
            self.tracker.set_run_id(run_id)
            self.tracker.initialize_run()

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

    def log_metrics(self, step, **metrics):
        metrics = {k: tensor2num(v) for k, v in metrics.items()}

        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                fig = plt_cmap(v)
                self.tracker.log_figures((k, fig))
                del metrics[k]

        if self.ctx.is_main_process:
            self.tracker.log_metrics(step, **metrics)

    def log_params(self, **params):
        if self.ctx.is_main_process:
            self.tracker.log_params(**params)

    def log_artifacts(self, *paths):
        if self.ctx.is_main_process:
            self.tracker.log_artifacts(*paths)

    def set_tags(self, **tags):
        self.tracker.set_tags(**tags)

    def check_run_status(self, run_id):
        return self.tracker.check_run_status(run_id)

    def find_similar_run(self):
        list_runs = self.tracker.list_existing_runs()
        for r in list_runs:
            run = self.tracker.get_run(r.run_id)
            run_name = run.data.tags[MLFLOW_RUN_NAME]
            if run_name == self.config['Manager']['run']:
                return r.run_id
        return False

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
    def __init__(self, config, run_id=None, trial=None):
        super(Experiment, self).__init__(config, run_id)

        self.batch_size = self.config['Training']['batch_size'] // self.world_size
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.partial_optimizer = None
        self.partial_lr_scheduler = None

        self.tracked_metric = self.config.get('Validation', {}).get('reference_metric', None)
        self.class_weights = None

        self.saved_models = {'best_valid': None, 'last': None}

        self.save_last = True
        self.run_training = True
        self.validation_batch_size = self.config.get('Validation', {}).get('batch_size', None)

        self.additional_datasets = {}
        self.additional_datasets_batch_size = {}

        self.data_keys = ['image']
        self._trial = trial

    def get_model_on_device(self, rank: int) -> nn.Module:
        tqdm.write('Rank %i, gpu %i' % (rank, self.get_gpu_from_rank(rank)))
        torch.cuda.set_device(self.get_gpu_from_rank(rank))
        model = self.get_model()
        model = self.convert_batch_norm(model)
        model = model.cuda(self.get_gpu_from_rank(rank))
        if self.multi_gpu:
            model = DDP(model, device_ids=[self.get_gpu_from_rank(rank)],
                        find_unused_parameters=self.ddp_config.get('find_unused_parameters', False))
        return model

    def get_model(self) -> nn.Module:
        assert self.model is not None, "The model has not been configured, call set_model(model) first"
        if self.continue_training:
            self.model.load(self.tracker.network_savepoint,
                            load_most_recent=True, map_location='cpu')

        return self.model

    def set_model(self, model: nn.Module) -> nn.Module:
        model = nnt_format(model)
        self.model = model

    def init_model(self):
        pass

    def set_params_group(self, params_group: dict):
        self.model.set_params_group(params_group)

    def batch_to_device(self, batch, rank: int):
        device = self.get_gpu_from_rank(rank)
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [b.cuda(device) if isinstance(
                b, torch.Tensor) else b for b in batch]
        elif isinstance(batch, dict):
            batch = {k: b.cuda(device) for k, b in batch.items()
                     if isinstance(b, torch.Tensor)}
        elif isinstance(batch, torch.Tensor):
            batch = batch.cuda(device)
        return batch

    def initial_tracking(self):
        self.log_params(**self.config.tracked_params)
        save_yaml(self.config, os.path.join(self.tracker.config_savepoint, 'config.yaml'))
        self.log_artifacts(self.config.get_path())

    def set_train_dataset(self, dataset):
        dataset = concat_datasets_if_needed(dataset)
        self.train_dataset = dataset

    def set_valid_dataset(self, dataset):
        dataset = concat_datasets_if_needed(dataset)
        self.validation_dataset = dataset

    def set_test_dataset(self, dataset):
        self.test_dataset = dataset

    def get_state_metric(self):
        if self.tracked_metric:
            return {'epoch': self.current_epoch,
                    self.tracked_metric: self.tracker.get_best_score_for_metric(self.tracked_metric)}

    def create_optimizer(self, **config):
        solver = config['solver']
        func = OPTIMS[solver]
        return partial_fill_kwargs(func, config['params_solver'])

    def register_dataset(self, dataset, name, batch_size=None):
        self.additional_datasets[name] = concat_datasets_if_needed(dataset)
        if batch_size is None:
            batch_size = self.batch_size
        self.additional_datasets_batch_size[name] = batch_size

    def set_optimizer(self, optimizers=None, **config):
        """
        :return: A partial function of an optimizer. Partially passed arguments are hyper parameters
        """
        if optimizers is not None:
            self.partial_optimizer = optimizers
        else:
            self.partial_optimizer = self.create_optimizer(**config)

    def set_scheduler(self):
        config = self.config.get('Learning_rate_scheduler', None)
        if config is not None:
            scheduler_name = config['scheduler']
            scheduler = SCHEDULERS[scheduler_name]
            self.ctx.scheduler_opt = scheduler
            self.ctx.scheduler_call_on = config.get(
                'update_type', self.ctx.scheduler_call_on)
            self.partial_lr_scheduler = partial_fill_kwargs(
                scheduler.func, config['params_scheduler'])

    def get_dataloader(self, dataset, shuffle=True,
                       batch_size=None,
                       num_workers=None,
                       drop_last=False,
                       persistent_workers=False,
                       rank=0):

        num_workers = self.config['Manager']['num_workers'] if num_workers is None else num_workers
        persistent_workers = persistent_workers | self.config['Manager'].get('persistent_workers', False)

        exp_dataloader = self.config['Manager'].get(
            'experimental_dataloader', False)

        c_shuffle = self.config['Dataset'].get('shuffle', True)
        shuffle = shuffle & c_shuffle

        batch_size = self.batch_size if batch_size is None else batch_size
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
                                                     persistent_workers=persistent_workers)
        else:
            dataloader = MultiEpochsDataLoader(dataset, batch_size=batch_size,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               shuffle=shuffle if sampler is None else False,
                                               sampler=sampler,
                                               worker_init_fn=set_non_torch_seed, drop_last=drop_last,
                                               persistent_workers=persistent_workers)

        return dataloader, sampler

    def save_model(self, model, filename, **kwargs):
        if not self.ctx.is_main_process:
            return
        tqdm.write('Saving model')
        save = model.save(
            savepoint=self.tracker.network_savepoint, filename=filename, **kwargs)

        if 'best_valid' in filename:
            self.saved_models['best_valid'] = save
        else:
            self.saved_models['last'] = save

        if self.config['Manager']['max_saved_model']:
            files = glob.glob(
                self.tracker.network_savepoint + "/best_valid_*.pth")
            files.sort(key=os.path.getmtime)
            for f in files[:-self.config['Manager']['max_saved_model']]:
                os.remove(f)

    def save_scripted_model(self, model):
        model.save_scripted(
            savepoint=self.tracker.network_savepoint, filename='model_scripted')

    def _start_process(self, rank: int = 0):
        tqdm.write('Initializing process %i' % rank)
        if self.multi_gpu:
            dist.init_process_group(backend=self.config['Manager']['dist_backend'], rank=rank,
                                    world_size=self.world_size)
        self.init_model()
        model = self.get_model_on_device(rank)
        self.ctx.rank = rank

        if self._trial and self.multi_gpu:
            if self.ctx.is_main_process:
                self._trial = TorchDistributedTrial(self._trial, self.ctx.rank)
            else:
                self._trial = TorchDistributedTrial(None, self.ctx.rank)

        if self.ctx.is_main_process and self.save_jit_model:
            self.save_scripted_model(model)
        if self.run_training:
            try:
                self.train(model, rank)
            except KeyboardInterrupt:
                if self.ctx.is_main_process:
                    Log.warn(
                        "Killed Process. The last model will be registered at %s" % self.saved_models)
                    self.tracker.set_status('KILLED')
            finally:
                self.tracker.set_status('FAILED')

        if self.ctx.is_main_process and (self.run_training or self.save_last):
            self.save_model(model, 'last')
            self.register_trained_model()

        if self.call_end_function:
            model._metrics.reset()
            self.end(model)

        self.clean_up()

    def start(self, run_id=None):
        if not self.run_started:
            self.start_run(run_id)

        if self.run_training:
            self.set_scheduler()
            assert self.partial_optimizer is not None, "Missing optimizer for training"
            assert self.train_dataset is not None, "Missing dataset"
            if self.validation_dataset is None:
                Log.warn(
                    "Missing validation set, default behaviour is to save the model once per epoch")

            if self.partial_lr_scheduler is None:
                Log.warn(
                    "Missing learning rate scheduler, default behaviour is to keep the learning rate constant")

        self.keyboard_exception_raised = False

        if self.register_params:
            self.initial_tracking()

        if self.multi_gpu:
            mp.spawn(self._start_process, nprocs=self.world_size, join=True)

        else:
            self._start_process()

        if not self.keyboard_exception_raised:
            self.tracker.set_status(status='FINISHED')

    def convert_batch_norm(self, model: nn.Module) -> nn.Module:
        if self.c['Network']['synchronized_batch_norm'] and self.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def register_trained_model(self):
        if self.saved_models['best_valid']:
            self.log_artifacts(self.saved_models['best_valid'])
        if self.saved_models['last']:
            self.log_artifacts(self.saved_models['last'])

    def lr_scheduler_step(self, validation_metrics=None):
        if self.ctx.lr_scheduler is None:
            return
        else:
            if validation_metrics is None:
                self.ctx.lr_scheduler.step()
            else:
                self.ctx.lr_scheduler.step(validation_metrics)

    def eval(self, register_params=False, run_id=None):
        self.register_params = register_params
        self.continue_training = False
        self.run_training = False
        self.start(run_id=run_id)

    def create_default_dataloader(self):
        rank = self.ctx.rank
        train_loader, train_sampler = self.get_dataloader(
            self.train_dataset, drop_last=True, rank=rank)
        for key, value in self.additional_datasets.items():
            self.ctx.additional_dataloader[key] = self.get_dataloader(value,
                                                                      drop_last=True,
                                                                      rank=rank,
                                                                      batch_size=self.additional_datasets_batch_size[
                                                                          key])

        self.ctx.train_loader = train_loader
        self.ctx.train_sampler = train_sampler

        if self.validation_dataset is not None:
            valid_loader, valid_sampler = self.get_dataloader(self.validation_dataset,
                                                              batch_size=self.world_size
                                                              if self.validation_batch_size is None else
                                                              self.validation_batch_size,
                                                              shuffle=True, rank=rank)
            self.ctx.valid_loader = valid_loader
            self.ctx.valid_sampler = valid_sampler

    def train(self, model, rank):
        optimizer = self.partial_optimizer(
            model.get_trainable_parameters())

        if self.partial_lr_scheduler is not None:
            lr_scheduler = self.partial_lr_scheduler(optimizer)
        else:
            lr_scheduler = None
        scaler = GradScaler(
            enabled=self.c['Manager'].get('grad_scaling', False))

        self.create_default_dataloader()
        self.ctx.lr_scheduler = lr_scheduler
        self.ctx.scaler = scaler
        self.ctx.optimizer = optimizer
        if self.run_profiling:
            self.profile(model)

        self.main_training_loop(model=model)

    def profile(self, model):
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=self.config['Manager']['num_workers'] + 1,
                                             repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile/' + self.tracker.run_id),
            record_shapes=True,
            profile_memory=True, with_flops=True,
            with_stack=True)
        prof.start()
        for step, batch in enumerate(self.ctx.train_loader):
            if step >= (2 + 2 + 10) * 2:
                break
            batch = self.batch_to_device(batch, rank=self.ctx.rank)
            loss = self.forward_train(model, self.loss, batch)
            self.ctx.scaler.scale(loss).backward()
            self.ctx.scaler.step(self.ctx.optimizer)
            self.ctx.scaler.update()
            model.zero_grad()
            prof.step()
        prof.stop()

    def end(self, model):
        pass

    def in_epoch(self, *args, **kwargs):
        pass

    def main_training_loop(self, model):
        total_epoch = self.config['Training'].get('epochs', -1)
        max_iterations = self.config['Training'].get('iterations', -1)
        assert (total_epoch > 0) or (max_iterations > 0), "You must define a number of training iterations or a number " \
                                                          "of epochs"
        if max_iterations > 0:
            total_epoch = math.ceil(max_iterations / self.ctx.epoch_size)
        # Reset epoch count from the saved iterations.

        for e in range(total_epoch):
            self.ctx.init_progress_bar()
            if self.ctx.train_sampler is not None:
                self.ctx.train_sampler.set_epoch(e)

            if self.ctx.is_main_process:
                tqdm.write('** Epoch %i/%i **' % (e, total_epoch))
                self.log_metrics(e, progress=100 * e / total_epoch)

            self.in_epoch(model=model)

        self.ctx.close_progress_bar()

    @property
    def metrics(self):
        return self.tracker.last_metrics()

    @property
    def current_iteration(self):
        return self.tracker.current_iteration

    @property
    def current_epoch(self):
        return math.floor(self.current_iteration / self.ctx.epoch_size)

    @current_iteration.setter
    def current_iteration(self, value: int):
        self.tracker.current_iteration = value

    def update_scheduler_on_epoch(self):
        if self.ctx.scheduler_call_on == 'on_epoch':
            self.lr_scheduler_step()

    def update_scheduler_on_validation(self, validation_metric):
        if self.ctx.scheduler_call_on == 'on_validation':
            self.lr_scheduler_step(validation_metric)

    def update_scheduler_on_iteration(self):
        if self.ctx.scheduler_call_on == 'on_iteration':
            self.lr_scheduler_step()

    def visualization_images(self, images, masks=None,
                             folder=None,
                             filename=None,
                             colors=None,
                             alpha=0.8):
        from torchvision.io import write_jpeg
        images = create_mosaic(images, masks, alpha=alpha, colors=colors)
        if filename is None:
            filename = 'image'
        if folder is None:
            folder = self.tracker.validation
        filepath = os.path.join(folder, filename + '.jpeg')
        write_jpeg(images, filepath)
        self.log_artifacts(filepath)

    def load_best_model(self, run_id=None):
        if not run_id:
            run_id = self.tracker.run_id
        if not self.run_started:
            self.start_run(run_id)

        self.init_model()
        self.model.load(self.tracker.network_savepoint,
                        filtername='best', load_most_recent=True, map_location='cpu')


@dataclass
class Context:
    train_loader = None
    train_sampler = None
    valid_loader = None
    valid_sampler = None
    loss_function = None
    lr_scheduler = None
    scaler = None
    optimizer = None
    scheduler_opt = None
    additional_dataloader: dict
    rank: int = 0
    rank_main_process: int = 0
    progress_bar = None
    multi_gpu: bool = False
    scheduler_call_on = 'on_epoch'
    _epoch_size = None

    @property
    def epoch_size(self):
        if self._epoch_size is None:
            return len(self.train_loader)
        else:
            return self._epoch_size

    @epoch_size.setter
    def epoch_size(self, value):
        self._epoch_size = value

    def init_progress_bar(self):
        if self.is_main_process:
            if self.progress_bar is None:
                self.progress_bar = tqdm(total=self.epoch_size)
            else:
                self.progress_bar.total = self.epoch_size
                self.progress_bar.refresh()
                self.progress_bar.reset()

    def update_progress_bar(self):
        if self.is_main_process:
            self.progress_bar.update(1)

    def close_progress_bar(self):
        if self.is_main_process:
            self.progress_bar.refresh()
            self.progress_bar.close()

    @property
    def is_main_process(self):
        return (self.rank == self.rank_main_process) or (not self.multi_gpu)

    def get_dataloader(self, name):
        return self.additional_dataloader[name]
