import glob
import os
import time
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm
from mlflow.tracking.client import MlflowClient
from torch.cuda.amp import autocast, GradScaler
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from nntools.dataset import get_class_count, class_weighting
from nntools.nnet.loss import FuseLoss, DiceLoss
from nntools.tracker import Tracker
from nntools.utils.io import create_folder, save_yaml
from nntools.utils.misc import convert_function
from nntools.utils.random import set_seed, set_non_torch_seed
from nntools.utils.torch import DistributedDataParallelWithAttributes as DDP

class Manager(ABC):
    def __init__(self, config):
        self.config = config
        self.seed = self.config['Manager']['seed']
        set_seed(self.seed)
        self.run_folder = os.path.join(self.config['Manager']['save_point'],
                                       self.config['Manager']['experiment'],
                                       self.config['Manager']['run'])
        create_folder(self.run_folder)
        self.network_savepoint = os.path.join(self.run_folder, 'trained_model')
        self.prediction_savepoint = os.path.join(self.run_folder, 'predictions')

        self.gpu = self.config['Manager']['gpu']

        if not isinstance(self.gpu, list):
            self.gpu = [self.gpu]

        self.world_size = len(self.config['Manager']['gpu'])
        self.multi_gpu = self.world_size > 1
        if self.multi_gpu:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'

        self.run_id = None

        self.mlflow_client = MlflowClient(os.path.join(self.config['Manager']['save_point'], 'mlruns'))
        exp_name = self.config['Manager']['experiment']

        exp = self.mlflow_client.get_experiment_by_name(exp_name)
        if exp is None:
            self.exp_id = self.mlflow_client.create_experiment(exp_name)
        else:
            self.exp_id = exp.experiment_id

    def convert_batch_norm(self, model):
        if self.config['CNN']['synchronized_batch_norm'] and self.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def start_run(self, run_id=None):
        tags = {MLFLOW_RUN_NAME: self.config['Manager']['run']}
        if run_id is None:
            run = self.mlflow_client.create_run(experiment_id=self.exp_id, tags=tags)
        else:
            run = self.mlflow_client.get_run(run_id=run_id)

        self.run_id = run.info.run_id
        # Update the save point in an unique way by using the run id
        self.network_savepoint = os.path.join(self.network_savepoint, str(self.run_id))
        self.prediction_savepoint = os.path.join(self.prediction_savepoint, str(self.run_id))
        create_folder(self.network_savepoint)
        create_folder(self.prediction_savepoint)

    def clean_up(self):
        if self.multi_gpu:
            dist.destroy_process_group()

    def is_main_process(self, rank):
        return rank == 0 or not self.multi_gpu

    def log_params(self, **params):
        for k, v in params.items():
            self.mlflow_client.log_param(self.run_id, k, v)

    def log_metrics(self, step, **metrics):
        for k, v in metrics.items():
            self.mlflow_client.log_metric(self.run_id, k, v, int(time.time() * 1000), step=step)

    def log_artifact(self, path):
        self.mlflow_client.log_artifact(self.run_id, path)


class Experiment(Manager):
    def __init__(self, config):
        super(Experiment, self).__init__(config)

        self.ignore_index = self.config['Training']['ignore_index'] if 'ignore_index' in self.config[
            'Training'] else -100
        self.batch_size = self.config['Training']['batch_size'] // self.world_size
        self.n_classes = config['CNN']['n_classes']

        self.dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.partial_optimizer = None
        self.partial_lr_scheduler = None
        self.tracked_metric = None
        self.class_weights = None
        self.model = None
        self.last_save = None
        self.run_training = True

    def set_train_dataset(self, dataset):
        self.dataset = dataset

    def set_valid_dataset(self, dataset):
        self.validation_dataset = dataset

    def set_test_dataset(self, dataset):
        self.test_dataset = dataset

    def set_optimizer(self, func, **kwargs):
        """
        :return: A partial function of an optimizers. Partial passed arguments are hyperparameters
        """
        self.partial_optimizer = convert_function(func, kwargs)

    def set_scheduler(self, func, **kwargs):
        self.partial_lr_scheduler = convert_function(func, kwargs)

    def set_model(self, model):
        self.model = model

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
        loss = FuseLoss()
        loss_args = self.config['Training']['segmentation_losses'].lower()
        if 'ce' in loss_args:
            if weights is None:
                loss.append(nn.CrossEntropyLoss(ignore_index=self.ignore_index))
            else:
                loss.append(nn.CrossEntropyLoss(weight=weights.cuda(rank),
                                                ignore_index=self.ignore_index))
        if 'dice' in loss_args:
            loss.append(DiceLoss(ignore_index=self.ignore_index))
        return loss

    def get_class_weights(self):
        class_count = get_class_count(self.dataset, save=True, load=True)
        return torch.tensor(class_weighting(class_count, mode=self.config['Training']['weighting_function'],
                                            ignore_index=self.ignore_index))

    def get_model(self):
        assert self.model is not None, "The model has not been configured, call set_model(model)"
        return self.model

    def setup_class_weights(self, weights):
        self.class_weights = weights

    def save_model(self, model, filename, **kwargs):
        self.last_save = model.save(savepoint=self.network_savepoint, filename=filename, **kwargs)
        if self.config['Manager']['max_saved_model']:
            files = glob.glob(self.network_savepoint + "/*.pth")
            files.sort(key=os.path.getmtime)
            for f in files[:-self.config['Manager']['max_saved_model']]:
                os.remove(f)

    def _start_process(self, rank=0):
        try:
            model = self.get_model_on_device(rank)
            if self.run_training:
                self.train(model, rank)
        except KeyboardInterrupt:
            Tracker.warn("Attempt to register model at %s" % self.last_save)
            if self.is_main_process(rank):
                self.register_trained_model()
                self.mlflow_client.set_terminated(self.run_id, status='KILLED')
            self.clean_up()
            raise KeyboardInterrupt
        except:
            if self.is_main_process(rank):
                self.mlflow_client.set_terminated(self.run_id, status='FAILED')
            self.clean_up()
            raise

        if self.is_main_process(rank):
            self.register_trained_model()
        dist.barrier()
        self.end(model, rank)
        self.clean_up()

    def get_model_on_device(self, rank):
        torch.cuda.set_device(rank)
        model = self.get_model()
        model = self.convert_batch_norm(model)
        model = model.cuda(rank)
        if self.multi_gpu:
            dist.init_process_group(self.config['Manager']['dist_backend'], rank=rank, world_size=self.world_size)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        return model

    def initial_tracking(self):
        self.log_params(**self.config['Training'])
        self.log_params(**self.config['Optimizer'])
        self.log_params(**self.config['Learning_rate_scheduler'])
        self.log_params(**self.config['CNN'])
        self.log_params(**self.config['Preprocessing'])

    def start(self):
        assert self.partial_optimizer is not None, "Missing optimizer for training"
        assert self.dataset is not None, "Missing dataset"

        if self.validation_dataset is None:
            Tracker.warn("Missing validation set, default behaviour is to save the model once per epoch")

        if self.partial_lr_scheduler is None:
            Tracker.warn("Missing learning rate scheduler, default behaviour is to keep the learning rate constant")

        if self.config['Training']['weighted_loss'] and self.class_weights is None:
            class_weights = self.get_class_weights()
            self.setup_class_weights(weights=class_weights)

        if self.run_id is None:
            Tracker.warn("Starting a new run")
            self.start_run()

        self.initial_tracking()
        if self.multi_gpu:
            mp.spawn(self._start_process,
                     nprocs=self.world_size,
                     join=True)
        else:
            self._start_process(rank=self.gpu[0])

        self.mlflow_client.set_terminated(self.run_id, status='FINISHED')
        save_yaml(self.config, os.path.join(self.run_folder, 'config.yaml'))

    def register_trained_model(self):
        if self.last_save is not None:
            self.log_artifact(self.last_save)

    def end(self, model, rank):
        pass

    def train(self, model, rank=0):
        loss_function = self.get_loss(self.class_weights, rank=rank)
        optimizer = self.partial_optimizer(model.get_trainable_parameters(self.config['Optimizer']['params_solver']['lr']))

        if self.partial_lr_scheduler is not None:
            lr_scheduler = self.partial_lr_scheduler(optimizer)
        else:
            lr_scheduler = None
        iteration = 0
        train_loader, train_sampler = self.get_dataloader(self.dataset)
        iters_to_accumulate = self.config['Training']['iters_to_accumulate']
        scaler = GradScaler(enabled=self.config['Training']['grad_scaling'])

        for e in range(self.config['Training']['epochs']):
            if train_sampler is not None:
                train_sampler.set_epoch(e)

            if self.is_main_process(rank):
                print('** Epoch %i **' % e)
                progressBar = tqdm.tqdm(total=len(train_loader))

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
                            model.eval()
                            valid_metric = self.validate(model, iteration, rank)
                            model.train()
                    if self.is_main_process(rank):
                        self.log_metrics(iteration, trainining_loss=loss.item())

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

            # The learning rate scheduler is called once per epoch
            self.lr_scheduler_step(lr_scheduler, iteration, valid_metric)

            if self.is_main_process(rank):
                progressBar.close()

            if self.multi_gpu:
                dist.barrier()

    def lr_scheduler_step(self, lr_scheduler, iteration, validation_metrics=None):
        if lr_scheduler is None:
            return
        if self.config['Learning_rate_scheduler']['update_type'] == 'on_validation':
            assert validation_metrics is not None, "Missing validation score to update the learning rate sheduler. Check" \
                                                   "what the validate function returns"
            lr_scheduler.step(validation_metrics)
        else:
            lr_scheduler.step(iteration)

    def validate(self, model, iteration, rank=0):
        pass
