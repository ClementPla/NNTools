import os
from abc import ABC, abstractmethod
from functools import partial
import glob
import mlflow
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from nntools.dataset import SegmentationDataset
from nntools.experiment.utils import set_seed, set_non_torch_seed
from nntools.nnet.loss import FuseLoss, DiceLoss
from nntools.tracker import Tracker
from nntools.utils.io import save_config


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

        self.ignore_index = self.config['Dataset']['ignore_index'] if 'ignore_index' in self.config['Dataset'] else -100
        self.batch_size = self.config['Training']['batch_size'] // self.world_size
        self.n_classes = config['CNN']['n_classes']
        self.dataset = None

    def set_seed(self):
        seed = self.config['Manager']['seed']
        set_seed(seed)

    def configure_dataset(self):
        self.dataset = SegmentationDataset(**self.config['Dataset'])

    @abstractmethod
    def configure_network(self):
        pass

    @abstractmethod
    def start(self):
        pass

    def build_dataloader(self, dataset, shuffle=True):
        if self.multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            dataloader = torch.utils.data.Dataloader(dataset, batch_size=self.batch_size,
                                                     num_workers=self.config['Manager']['num_workers'],
                                                     pin_memory=True, sampler=sampler, worker_init_fn=set_non_torch_seed
                                                     )
        else:
            sampler = None
            dataloader = torch.utils.data.Dataloader(dataset, batch_size=self.batch_size,
                                                     num_workers=self.config['Manager']['num_workers'],
                                                     pin_memory=True, shuffle=shuffle, worker_init_fn=set_non_torch_seed)
        return dataloader, sampler


class Trainer(Manager):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        self.validation_dataset = None
        self.loss = None
        self.partial_optimizer = None
        self.tracked_metric = None

    def set_validation_dataset(self, dataset):
        self.validation_dataset = dataset

    def set_optimizer(self, func, **hyperparameters):
        """
        :return: A partial function of an optimizers. Partial passed arguments are hyperparameters
        """
        self.partial_optimizer = partial(func, **hyperparameters)

    def set_loss(self, weights=None):
        self.loss = FuseLoss()
        loss_args = self.config['Training']['segmentation_losses'].lower()

        if 'ce' in loss_args:
            self.loss.append(nn.CrossEntropyLoss(weight=weights,
                                                 ignore_index=self.ignore_index))

        if 'dice' in loss_args:
            self.loss.append(DiceLoss(ignore_index=self.ignore_index))

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

        optimizer = self.partial_optimizer(model.parameters())
        self.train(model, optimizer, rank)
        self.clean_up()

    def log_params(self):
        mlflow.log_params(self.config['Training'])
        mlflow.log_param("batch", self.config['Dataset']['batch_size'])
        mlflow.log_param("resolution", self.config['Dataset']['img_size'])

    def log_metrics(self, step, **metrics):
        mlflow.log_metrics(metrics, step=step)

    def start(self):
        assert self.loss is not None, "Missing loss function for training, call set_loss() on trainer"
        assert self.partial_optimizer is not None, "Missing optimizer for training"
        assert self.dataset is not None, "Missing dataset"
        if self.validation_dataset is None:
            Tracker.warn("Missing validation set, default behaviour for model saving is once per epoch")

        mlflow.set_experiment(self.config['Manager']['experiment'])
        with mlflow.start_run(run_name=self.config['Manager']['run']):
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

    def train(self, model, optimizer, rank=0):
        from torch.cuda.amp import autocast, GradScaler
        train_loader, train_sampler = self.build_dataloader(self.dataset)
        iters_to_accumulate = self.config['Training']['iters_to_accumulate']
        scaler = GradScaler(enabled=self.config['Manager']['grad_scaling'])

        for e in self.config['Training']['epochs']:
            if train_sampler is not None:
                train_sampler.set_epoch(e)

            for i, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                img = batch[0].cuda(rank)
                gt = batch[1].cuda(rank)
                with autocast(enabled=self.config['Manager']['amp']):
                    pred = model(img)
                    loss = self.loss(pred, gt) / iters_to_accumulate

                scaler.scale(loss).backward()
                if (i + 1) % iters_to_accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                iteration = i + e * len(train_loader)
                if iteration % self.config['Validation']['log_interval']:
                    if self.validation_dataset is not None:
                        with torch.no_grads():
                            self.validate(model, iteration, rank)

            if self.validation_dataset is None:
                self.save_model(model, filename='iteration_%i_loss_%f' % (iteration, loss.item()))

    @abstractmethod
    def validate(self, model, iteration, rank=0):
        pass
