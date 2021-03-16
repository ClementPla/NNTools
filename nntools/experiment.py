import datetime
import glob
import os
from abc import ABC

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

import torch.nn as nn
import tqdm
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from nntools.dataset import class_weighting
from nntools.nnet import nnt_format, FuseLoss, SUPPORTED_LOSS, BINARY_MODE, MULTICLASS_MODE
from nntools.tracker import Log, Tracker, log_params, log_metrics, log_artifact
from nntools.utils.io import save_yaml
from nntools.utils.misc import partial_fill_kwargs, call_with_filtered_kwargs
from nntools.utils.optims import OPTIMS
from nntools.utils.random import set_seed, set_non_torch_seed
from nntools.utils.scheduler import SCHEDULERS
from nntools.utils.torch import DistributedDataParallelWithAttributes as DDP


class Manager(ABC):
    def __init__(self, config, run_id=None):
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
        self.register_params = True
        self.call_end_function = True
        self.keyboard_exception_raised = False
        self.run_started = False
        if not isinstance(self.gpu, list):
            self.gpu = [self.gpu]

        self.world_size = len(self.gpu)
        self.multi_gpu = self.world_size > 1

        if self.multi_gpu:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

    def convert_batch_norm(self, model):
        if self.config['Network']['synchronized_batch_norm'] and self.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def start_run(self, run_id=None):
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
        self.run_started = True

    def clean_up(self):
        if self.multi_gpu:
            dist.destroy_process_group()

    def is_main_process(self, rank):
        return (rank == 0) or not self.multi_gpu

    def get_gpu_from_rank(self, rank):
        return self.gpu[rank]

    def get_model_on_device(self, rank):
        print('Rank %i, gpu %i' % (rank, self.get_gpu_from_rank(rank)))
        torch.cuda.set_device(self.get_gpu_from_rank(rank))
        model = self.get_model()
        model = self.convert_batch_norm(model)
        model = model.cuda(self.get_gpu_from_rank(rank))
        if self.multi_gpu:
            model = DDP(model, device_ids=[self.get_gpu_from_rank(rank)], find_unused_parameters=True)
        return model

    def get_model(self):
        assert self.model is not None, "The model has not been configured, call set_model(model)"
        if self.continue_training:
            self.model.load(self.tracker.network_savepoint, load_most_recent=True, map_location='cpu')

        return self.model

    def set_model(self, model):
        model = nnt_format(model)
        self.model = model
        return model

    def set_params_group(self, params_group):
        self.model.set_params_group(params_group)

    def batch_to_device(self, batch, rank):
        device = self.get_gpu_from_rank(rank)
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [b.cuda(device) if isinstance(b, torch.Tensor) else b for b in batch]
        else:
            batch = batch.cuda(device)
        return batch


class Experiment(Manager):
    def __init__(self, config, run_id=None):
        super(Experiment, self).__init__(config, run_id)

        self.ignore_index = self.config['Training']['ignore_index'] \
            if 'ignore_index' in self.config['Training'] \
            else -100

        self.batch_size = self.config['Training']['batch_size'] // self.world_size
        self.n_classes = config['Network']['n_classes']

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
        log_params(self.tracker, **self.config['Training'])
        log_params(self.tracker, **self.config['Optimizer'])
        log_params(self.tracker, **self.config['Learning_rate_scheduler'])
        log_params(self.tracker, **self.config['Network'])
        log_params(self.tracker, Loss=self.config['Loss']['type'])
        if 'fusion' in self.config['Loss']:
            log_params(self.tracker, Loss_fusion=self.config['Loss']['fusion'])

        if 'params_loss' in self.config['Loss']:
            log_params(self.tracker, **self.config['Loss']['params_loss'])

        log_params(self.tracker, weighted_loss=self.config['Loss']['weighted_loss'])
        if 'params_weighting' in self.config['Loss'] and self.config['Loss'].get('weighted_loss', False):
            log_params(self.tracker, **self.config['Loss']['params_weighting'])

        log_params(self.tracker, **self.config['Preprocessing'])
        log_artifact(self.tracker, self.config.get_path())

    def set_train_dataset(self, dataset):
        self.train_dataset = dataset

    def set_valid_dataset(self, dataset):
        self.validation_dataset = dataset

    def set_test_dataset(self, dataset):
        self.test_dataset = dataset

    def set_optimizer(self, **config):
        """
        :return: A partial function of an optimizers. Partial passed arguments are hyperparameters
        """
        solver = config['solver']
        func = OPTIMS[solver]
        self.partial_optimizer = partial_fill_kwargs(func, config['params_solver'])

    def set_scheduler(self, **config):
        scheduler_name = config['scheduler']
        scheduler = SCHEDULERS[scheduler_name]
        self.ctx_train['scheduler_opt'] = scheduler
        self.partial_lr_scheduler = partial_fill_kwargs(scheduler.func, config['params_scheduler'])

    def get_dataloader(self, dataset, shuffle=True, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                     num_workers=self.config['Manager']['num_workers'],
                                                     pin_memory=True, sampler=sampler, worker_init_fn=set_non_torch_seed
                                                     )
        else:
            sampler = None
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                     num_workers=self.config['Manager']['num_workers'],
                                                     pin_memory=True, shuffle=shuffle,
                                                     worker_init_fn=set_non_torch_seed)
        return dataloader, sampler

    def get_loss(self, weights=None, rank=0):
        config = self.config['Loss']
        fuse_loss = FuseLoss(fusion=config.get('fusion', 'mean'))
        mode = MULTICLASS_MODE if self.n_classes > 2 else BINARY_MODE

        list_losses = config['type'].split('|')

        if weights is not None:
            weights = weights.cuda(self.get_gpu_from_rank(rank))
        kwargs = {'ignore_index': self.ignore_index, 'mode': mode}

        for k in list_losses:
            k = k.strip()
            loss = SUPPORTED_LOSS[k]
            loss_args = kwargs.copy()
            loss_args['weight'] = weights

            if k in config.get('params_loss', {}):
                loss_args.update(config['params_loss'][k])

            fuse_loss.add(call_with_filtered_kwargs(loss, loss_args))

        return fuse_loss

    def get_class_weights(self):
        class_count = self.train_dataset.get_class_count()
        kwargs = self.config['Loss'].get('params_weighting', {})
        return torch.tensor(class_weighting(class_count, ignore_index=self.ignore_index, **kwargs))

    def setup_class_weights(self, weights):
        if self.config['Manager']['amp']:
            self.class_weights = weights.half()
        else:
            self.class_weights = weights

    def save_model(self, model, filename, **kwargs):
        print('Saving model')
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

    def _start_process(self, rank=0):
        if self.multi_gpu:
            dist.init_process_group(self.config['Manager']['dist_backend'], rank=rank, world_size=self.world_size)
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
                    Log.warn("Killed Process. The model will be registered at %s" % self.saved_models)
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

    def start(self):
        if not self.run_started:
            self.start_run()
        assert self.partial_optimizer is not None, "Missing optimizer for training"
        assert self.train_dataset is not None, "Missing dataset"

        self.keyboard_exception_raised = False
        if self.validation_dataset is None:
            Log.warn("Missing validation set, default behaviour is to save the model once per epoch")

        if self.partial_lr_scheduler is None:
            Log.warn("Missing learning rate scheduler, default behaviour is to keep the learning rate constant")

        if self.config['Loss']['weighted_loss'] and self.class_weights is None:
            class_weights = self.get_class_weights()
            self.setup_class_weights(weights=class_weights)

        if self.register_params:
            self.initial_tracking()

        if self.multi_gpu:
            mp.spawn(self._start_process,
                     nprocs=self.world_size,
                     join=True)

        else:
            self._start_process()

        if not self.keyboard_exception_raised:
            self.tracker.set_status(status='FINISHED')

        save_yaml(self.config, os.path.join(self.tracker.run_folder, 'config.yaml'))

    def register_trained_model(self):
        if self.saved_models['best_valid']:
            log_artifact(self.tracker, self.saved_models['best_valid'])
        if self.saved_models['last']:
            log_artifact(self.tracker, self.saved_models['last'])

    def end(self, model, rank):
        pass

    def forward_train(self, model, loss_function, rank, batch):
        batch = self.batch_to_device(batch, rank)
        pred = model(batch[0])
        if isinstance(pred, tuple):
            loss = loss_function(*pred, *batch[1:])
        else:
            loss = loss_function(pred, *batch[1:])
        return loss

    def train(self, model, rank=0):
        loss_function = self.get_loss(self.class_weights, rank=rank)
        optimizer = self.partial_optimizer(
            model.get_trainable_parameters(self.config['Optimizer']['params_solver']['lr']))

        if self.partial_lr_scheduler is not None:
            lr_scheduler = self.partial_lr_scheduler(optimizer)
        else:
            lr_scheduler = None
        iteration = self.tracker.current_iteration - 1
        train_loader, train_sampler = self.get_dataloader(self.train_dataset)
        iters_to_accumulate = self.config['Training'].get('iters_to_accumulate', 1)
        scaler = GradScaler(enabled=self.config['Manager'].get('grad_scaling', True))
        clip_grad = self.config['Training'].get('grad_clipping', False)

        for e in range(self.config['Training']['epochs']):
            if train_sampler is not None:
                train_sampler.set_epoch(e)

            if self.is_main_process(rank):
                print('** Epoch %i **' % e)
                progressBar = tqdm.tqdm(total=len(train_loader))

            for i, batch in (enumerate(train_loader)):
                iteration += 1
                with autocast(enabled=self.config['Manager']['amp']):
                    loss = self.forward_train(model, loss_function, rank, batch)

                loss = loss / iters_to_accumulate
                scaler.scale(loss).backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if clip_grad:
                        clip_grad_norm_(model.parameters(), float(clip_grad))
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()

                    if self.ctx_train['scheduler_opt'].call_on == 'on_iteration':
                        self.lr_scheduler_step(lr_scheduler, e, i, len(train_loader))

                """
                Validation step
                """
                if iteration % self.config['Validation']['log_interval'] == 0:
                    if self.validation_dataset is not None:
                        with torch.no_grad():
                            with autocast(enabled=self.config['Manager']['amp']):
                                valid_metric = self.validate(model, iteration, rank, loss_function)

                    if self.ctx_train['scheduler_opt'].call_on == 'on_validation':
                        self.lr_scheduler_step(lr_scheduler, e, i, len(train_loader), valid_metric)

                    if self.is_main_process(rank):
                        log_metrics(self.tracker, iteration, trainining_loss=loss.item())
                        self.save_model(model, filename='last')

                    if self.multi_gpu:
                        dist.barrier()

                if self.is_main_process(rank):
                    progressBar.update(1)

            """ 
            If the validation set is not provided, we save the model once per epoch
            """
            if self.validation_dataset is None:
                if self.is_main_process(rank):
                    self.save_model(model, filename='iteration_%i_loss_%f' % (iteration, loss.item()))

            if self.ctx_train['scheduler_opt'].call_on == 'on_epoch':
                self.lr_scheduler_step(lr_scheduler, e, iteration, len(train_loader))

            if self.is_main_process(rank):
                progressBar.close()

    def lr_scheduler_step(self, lr_scheduler, epoch, iteration, size_epoch, validation_metrics=None):
        if lr_scheduler is None:
            pass
        elif self.ctx_train['scheduler_opt'].call_on == 'on_validation':
            lr_scheduler.step(validation_metrics)
        else:
            lr_scheduler.step(self.ctx_train['scheduler_opt'].callback(epoch, iteration, size_epoch))

    def validate(self, model, iteration, rank=0, loss_function=None):
        pass
