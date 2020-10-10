import glob
import os
import time
from abc import ABC, abstractmethod

import mlflow
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm
from mlflow.tracking.client import MlflowClient
from torch.cuda.amp import autocast, GradScaler

from nntools.experiment.utils import set_seed, set_non_torch_seed
from nntools.nnet.loss import FuseLoss, DiceLoss
from nntools.tracker import Tracker
from nntools.utils.misc import convert_function
from nntools.utils.io import save_config
from nntools.utils.torch import DistributedDataParallelWithAttributes as DDP


class Manager(ABC):
    def __init__(self, config):
        self.config = config
        self.run_folder = os.path.join(self.config['Manager']['save_point'],
                                       self.config['Manager']['experiment'],
                                       self.config['Manager']['run'])
        self.network_savepoint = os.path.join(self.run_folder, 'trained_model')

        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)

        if not os.path.exists(self.network_savepoint):
            os.makedirs(self.network_savepoint)

        self.set_seed()
        self.world_size = len(self.config['Manager']['gpu'])
        self.multi_gpu = self.world_size > 1
        if self.multi_gpu:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

        self.ignore_index = self.config['Training']['ignore_index'] if 'ignore_index' in self.config[
            'Training'] else -100

        self.batch_size = self.config['Training']['batch_size'] // self.world_size
        self.n_classes = config['CNN']['n_classes']
        self.dataset = None

    def set_seed(self):
        seed = self.config['Manager']['seed']
        set_seed(seed)

    @abstractmethod
    def configure_network(self):
        pass

    @abstractmethod
    def start(self):
        pass

    def build_dataloader(self, dataset, shuffle=True):
        if self.multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                     num_workers=self.config['Manager']['num_workers'],
                                                     pin_memory=True, sampler=sampler, worker_init_fn=set_non_torch_seed
                                                     )
        else:
            sampler = None
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                     num_workers=self.config['Manager']['num_workers'],
                                                     pin_memory=True, shuffle=shuffle,
                                                     worker_init_fn=set_non_torch_seed)
        return dataloader, sampler


class Trainer(Manager):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        self.validation_dataset = None
        self.partial_optimizer = None
        self.partial_lr_scheduler = None
        self.tracked_metric = None
        self.run_id = -1
        self.weights = None

    def set_validation_dataset(self, dataset):
        self.validation_dataset = dataset

    def set_optimizer(self, func, **kwargs):
        """
        :return: A partial function of an optimizers. Partial passed arguments are hyperparameters
        """
        self.partial_optimizer = convert_function(func, kwargs)

    def set_scheduler(self, func, **kwargs):
        self.partial_lr_scheduler = convert_function(func, kwargs)

    def get_loss(self, weights=None, rank=0):
        loss = FuseLoss()
        loss_args = self.config['Training']['segmentation_losses'].lower()

        if 'ce' in loss_args:
            loss.append(nn.CrossEntropyLoss(weight=weights.cuda(rank),
                                            ignore_index=self.ignore_index))

        if 'dice' in loss_args:
            loss.append(DiceLoss(ignore_index=self.ignore_index))
        return loss

    def set_weights(self, weights):
        self.weights = weights

    def save_model(self, model, filename, **kwargs):
        model.save(savepoint=self.network_savepoint, filename=filename, **kwargs)
        if self.config['Manager']['max_saved_model']:
            files = glob.glob(self.network_savepoint + "/*.pth")
            files.sort(key=os.path.getmtime)
            for f in files[:-self.config['Manager']['max_saved_model']]:
                os.remove(f)

    def init_training(self, rank=0):
        torch.cuda.set_device(rank)
        model = self.configure_network()
        if self.config['CNN']['synchronized_batch_norm'] and self.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = model.cuda(rank)
        if self.multi_gpu:
            dist.init_process_group(self.config['Manager']['dist_backend'], rank=rank, world_size=self.world_size)
            model = DDP(model, device_ids=[rank])

        self.train(model, rank)
        self.clean_up()

    def log_params(self):
        mlflow.log_params(self.config['Training'])
        mlflow.log_params(self.config['Optimizer'])
        mlflow.log_params(self.config['Learning_rate_scheduler'])
        mlflow.log_params(self.config['CNN'])
        mlflow.log_params(self.config['Preprocessing'])

    def log_metrics(self, step, **metrics):
        for k, v in metrics.items():
            MlflowClient().log_metric(self.run_id, k, v, int(time.time() * 1000), step=step)

    def start(self):
        assert self.partial_optimizer is not None, "Missing optimizer for training"
        assert self.dataset is not None, "Missing dataset"
        if self.validation_dataset is None:
            Tracker.warn("Missing validation set, default behaviour for model saving is once per epoch")

        if self.partial_lr_scheduler is None:
            Tracker.warn("Missing learning rate scheduler, default behaviour is to keep the lr constant")

        mlflow.set_experiment(self.config['Manager']['experiment'])
        with mlflow.start_run(run_name=self.config['Manager']['run']):
            self.log_params()
            self.run_id = mlflow.active_run().info.run_id
            if self.multi_gpu:
                mp.spawn(self.init_training,
                         nprocs=self.world_size,
                         join=True)
            else:
                self.init_training(rank=self.gpu)
        save_config(self.config, os.path.join(self.run_folder, 'config.yaml'))

    def clean_up(self):
        if self.multi_gpu:
            dist.destroy_process_group()

    def train(self, model, rank=0):
        loss_function = self.get_loss(self.weights, rank=rank)
        optimizer = self.partial_optimizer(model.get_trainable_parameters(self.config['Optimizer']['lr']))
        if self.partial_lr_scheduler is not None:
            lr_scheduler = self.partial_lr_scheduler(optimizer)
        else:
            lr_scheduler = None

        train_loader, train_sampler = self.build_dataloader(self.dataset)
        iters_to_accumulate = self.config['Training']['iters_to_accumulate']
        scaler = GradScaler(enabled=self.config['Training']['grad_scaling'])

        for e in range(self.config['Training']['epochs']):
            if train_sampler is not None:
                train_sampler.set_epoch(e)

            if rank == 0 or not self.multi_gpu:
                print('** Epoch %i **' % e)
                t = tqdm.tqdm(total=len(train_loader))

            for i, batch in (enumerate(train_loader)):
                iteration = i + e * len(train_loader)
                img = batch[0].cuda(rank)
                gt = batch[1].cuda(rank)

                with autocast(enabled=self.config['Training']['amp']):
                    pred = model(img)
                    loss = loss_function(pred, gt) / iters_to_accumulate

                scaler.scale(loss).backward()
                if (i + 1) % iters_to_accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                """
                Validation step
                """
                if iteration % self.config['Validation']['log_interval'] == 0:
                    if self.validation_dataset is not None:
                        with torch.no_grad():
                            valid_metric = self.validate(model, iteration, rank)
                    if self.multi_gpu:
                        dist.barrier()

                    if rank == 0 or not self.multi_gpu:
                        self.log_metrics(iteration, trainining_loss=loss.item())

                if rank == 0 or not self.multi_gpu:
                    t.update(1)

            """ 
            If not validation set is provided, we save the model once per epoch
            """
            if self.validation_dataset is None:
                self.save_model(model, filename='iteration_%i_loss_%f' % (iteration, loss.item()))

            self.lr_scheduler_step(lr_scheduler, iteration, valid_metric)

            if rank == 0 or not self.multi_gpu:
                t.close()

    def lr_scheduler_step(self, lr_scheduler, iteration, validation_metrics=None):
        if lr_scheduler is None:
            return
        if self.config['Learning_rate_scheduler']['update_type'] == 'on_validation':
            assert validation_metrics is not None, "Missing validation score to update the learning rate sheduler. Check" \
                                                   "what the validate function returns"
            lr_scheduler.step(validation_metrics)
        else:
            lr_scheduler.step(iteration)

    @abstractmethod
    def validate(self, model, iteration, rank=0):
        pass
