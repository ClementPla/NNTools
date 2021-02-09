import glob
import os
import time
from abc import ABC

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm
import yaml
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from torch.cuda.amp import autocast, GradScaler

from nntools.dataset import class_weighting
from nntools.nnet import nnt_format
from nntools.nnet.loss import FuseLoss, SUPPORTED_LOSS, BINARY_MODE, MULTICLASS_MODE
from nntools.tracker import Log, Tracker, log_params, log_metrics, log_artifact
from nntools.utils.io import save_yaml
from nntools.utils.misc import partial_fill_kwargs
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
        self.tracker.create_client(os.path.join(self.config['Manager']['save_point'], 'mlruns'))
        self.tracker.set_run_folder(os.path.join(self.config['Manager']['save_point'],
                                                 self.config['Manager']['experiment'],
                                                 self.config['Manager']['run']))
        self.tracker.init_default_path()

        self.model = None
        self.gpu = self.config['Manager']['gpu']
        self.continue_training = True
        self.register_params = True

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


        tags = {MLFLOW_RUN_NAME: self.config['Manager']['run']}
        if run_id is None and self.tracker.run_id is None:
            self.tracker.create_run(tags)
        else:
            self.tracker.get_run(run_id)

        if self.continue_training:
            self.tracker.go_to_exp_last_iteration()

        self.tracker.init_default_path()
        self.tracker.set_status('RUNNING')
        Log.warn('Run started (status = RUNNING)')

    def clean_up(self):
        if self.multi_gpu:
            dist.destroy_process_group()

    def is_main_process(self, rank):
        return rank == 0 or not self.multi_gpu

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
            self.model.load(self.tracker.network_savepoint, load_most_recent=True)

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
            batch = (b.cuda(device) for b in batch)
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

        self.last_save = {'best_valid': None,
                          'last': None}

        self.run_training = True
        self.ctx_train = {}

    def initial_tracking(self):
        log_params(self.tracker, **self.config['Training'])
        log_params(self.tracker, **self.config['Optimizer'])
        log_params(self.tracker, **self.config['Learning_rate_scheduler'])
        log_params(self.tracker, **self.config['Network'])
        log_params(self.tracker, **self.config['Loss'])
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
        solver = config.pop('solver')
        func = OPTIMS[solver]
        self.partial_optimizer = partial_fill_kwargs(func, config['params_solver'])

    def set_scheduler(self, **config):
        scheduler_name = config.pop('scheduler')
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
        fuse_loss = FuseLoss(fusion=config.pop('fusion', 'mean'))
        mode = MULTICLASS_MODE if self.n_classes > 2 else BINARY_MODE

        list_losses = [k for k in config.keys()]
        existing_losses = [l for l in SUPPORTED_LOSS.keys()]
        for k in list_losses:
            if k not in existing_losses:
                raise NotImplementedError("Loss %s is not implemented" % k)
            kwargs = {'ignore_index': self.ignore_index}
            if k in ['Dice', 'Focal', 'Jaccard', 'Lovasz']:
                kwargs['mode'] = mode
            if k in ['CrossEntropy', 'SoftBinaryCrossEntropy'] and weights is not None:
                kwargs['weight'] = weights.cuda(self.get_gpu_from_rank(rank))
            if config[k] is not None:
                kwargs.update(config[k])
            fuse_loss.add(SUPPORTED_LOSS[k](**kwargs))

        return fuse_loss

    def get_class_weights(self):
        class_count = self.train_dataset.get_class_count()
        kwargs = self.config['Training'].pop('params_weighting', {})
        return torch.tensor(class_weighting(class_count, ignore_index=self.ignore_index, **kwargs))

    def setup_class_weights(self, weights):
        self.class_weights = weights

    def save_model(self, model, filename, **kwargs):
        save = model.save(savepoint=self.tracker.network_savepoint, filename=filename, **kwargs)
        if 'best_valid' in filename:
            self.last_save['best_valid'] = save
        else:
            self.last_save['last'] = save
        if self.config['Manager']['max_saved_model']:
            files = glob.glob(self.tracker.network_savepoint + "/best_valid_*.pth")
            files.sort(key=os.path.getmtime)
            for f in files[:-self.config['Manager']['max_saved_model']]:
                os.remove(f)

    def _start_process(self, rank=0):
        torch.cuda.set_device(rank)
        if self.multi_gpu:
            dist.init_process_group(self.config['Manager']['dist_backend'], rank=rank, world_size=self.world_size)
        try:
            model = self.get_model_on_device(rank)
            if self.run_training:
                with autocast(enabled=self.config['Training']['amp']):
                    self.train(model, rank)
        except KeyboardInterrupt:
            if self.is_main_process(rank):
                Log.warn("Killed Process. The model will be registered at %s" % self.last_save)
                self.save_model(model, 'last')
                self.register_trained_model()
                run_end = (input("Call end function ? [y]/n ") or "y").lower()
                self.tracker.set_status('KILLED')

                if run_end == 'y':
                    self.set_status_to_killed = True

                if not self.set_status_to_killed:
                    self.clean_up()
                    raise KeyboardInterrupt
        except:
            self.tracker.set_status('FAILED')
            self.clean_up()
            raise

        if self.multi_gpu:
            dist.barrier()

        if self.is_main_process(rank) and self.run_training and not self.set_status_to_killed:
            self.save_model(model, 'last')
            self.register_trained_model()

        with autocast(enabled=self.config['Training']['amp']):
            self.end(model, rank)

        if self.set_status_to_killed and self.is_main_process(rank):
            self.tracker.set_status(status='KILLED')
        self.clean_up()

    def start(self):
        assert self.partial_optimizer is not None, "Missing optimizer for training"
        assert self.train_dataset is not None, "Missing dataset"

        self.set_status_to_killed = False

        if self.validation_dataset is None:
            Tracker.warn("Missing validation set, default behaviour is to save the model once per epoch")

        if self.partial_lr_scheduler is None:
            Tracker.warn("Missing learning rate scheduler, default behaviour is to keep the learning rate constant")

        if self.config['Training']['weighted_loss'] and self.class_weights is None:
            class_weights = self.get_class_weights()
            self.setup_class_weights(weights=class_weights)

        self.start_run()

        if self.register_params:
            self.initial_tracking()

        if self.multi_gpu:
            mp.spawn(self._start_process,
                     nprocs=self.world_size,
                     join=True)
        else:
            self._start_process(rank=0)

        if not self.set_status_to_killed:
            self.tracker.set_status(status='FINISHED')
        save_yaml(self.config, os.path.join(self.tracker.run_folder, 'config.yaml'))

    def register_trained_model(self):
        if self.last_save['best_valid']:
            self.log_artifact(self.last_save['best_valid'])
        if self.last_save['last']:
            self.log_artifact(self.last_save['last'])

    def end(self, model, rank):
        pass

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
        iters_to_accumulate = self.config['Training']['iters_to_accumulate']
        scaler = GradScaler(enabled=self.config['Training']['grad_scaling'])

        for e in range(self.config['Training']['epochs']):
            if train_sampler is not None:
                train_sampler.set_epoch(e)

            if self.is_main_process(rank):
                print('** Epoch %i **' % e)
                progressBar = tqdm.tqdm(total=len(train_loader))

            for i, batch in (enumerate(train_loader)):
                iteration += 1
                img, gt = self.batch_to_device(batch, rank)
                pred = model(img)
                loss = loss_function(pred, gt) / iters_to_accumulate
                scaler.scale(loss).backward()
                if (i + 1) % iters_to_accumulate == 0:
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
                            valid_metric = self.validate(model, iteration, rank)
                            self.lr_scheduler_step(lr_scheduler, e, i, len(train_loader), valid_metric)

                    if self.is_main_process(rank):
                        self.log_metrics(iteration, trainining_loss=loss.item())
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

    def validate(self, model, iteration, rank=0):
        pass
