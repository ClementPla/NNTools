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
from mlflow.tracking.client import MlflowClient
from torch.cuda.amp import autocast, GradScaler
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from nntools.dataset import class_weighting
from nntools.nnet.loss import FuseLoss, SUPPORTED_LOSS, BINARY_MODE, MULTICLASS_MODE
from nntools.tracker import Tracker
from nntools.utils.io import create_folder, save_yaml
from nntools.utils.misc import partial_fill_kwargs
from nntools.utils.random import set_seed, set_non_torch_seed
from nntools.utils.scheduler import SCHEDULER
from nntools.utils.optims import OPTIMS
from nntools.utils.torch import DistributedDataParallelWithAttributes as DDP
from nntools.nnet import nnt_format


class Manager(ABC):
    def __init__(self, config, run_id=None):
        self.config = config
        self.seed = self.config['Manager']['seed']
        set_seed(self.seed)
        self.run_folder = os.path.join(self.config['Manager']['save_point'],
                                       self.config['Manager']['experiment'],
                                       self.config['Manager']['run'])
        create_folder(self.run_folder)
        self.network_savepoint = os.path.join(self.run_folder, 'trained_model')
        self.prediction_savepoint = os.path.join(self.run_folder, 'predictions')
        self.model = None
        self.current_iteration = 0
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

        self.run_id = run_id

        self.mlflow_client = MlflowClient(os.path.join(self.config['Manager']['save_point'], 'mlruns'))
        exp_name = self.config['Manager']['experiment']

        exp = self.mlflow_client.get_experiment_by_name(exp_name)
        if exp is None:
            self.exp_id = self.mlflow_client.create_experiment(exp_name)
        else:
            self.exp_id = exp.experiment_id

    def convert_batch_norm(self, model):
        if self.config['Network']['synchronized_batch_norm'] and self.multi_gpu:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def start_run(self, run_id=None):

        tags = {MLFLOW_RUN_NAME: self.config['Manager']['run']}
        if run_id is None and self.run_id is None:
            run = self.mlflow_client.create_run(experiment_id=self.exp_id, tags=tags)
            self.continue_training = False
        elif run_id is None:
            run = self.mlflow_client.get_run(run_id=self.run_id)
        elif self.run_id is None:
            run = self.mlflow_client.get_run(run_id=run_id)
        self.run_id = run.info.run_id
        self.mlflow_client.set_terminated(self.run_id, status='RUNNING')

        if self.continue_training:
            "Set the current iteration to the max iteration stored in the run"
            for k, v in run.data.metrics.items():
                his = self.mlflow_client.get_metric_history(self.run_id, k)
                self.current_iteration = max(self.current_iteration, his[-1].step)

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
            self.model.load(self.network_savepoint, load_most_recent=True)

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

        self.ignore_index = self.config['Training']['ignore_index'] if 'ignore_index' in self.config[
            'Training'] else -100
        self.batch_size = self.config['Training']['batch_size'] // self.world_size
        self.n_classes = config['Network']['n_classes']

        self.dataset = None
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
        self.log_params(**self.config['Training'])
        self.log_params(**self.config['Optimizer'])
        self.log_params(**self.config['Learning_rate_scheduler'])
        self.log_params(**self.config['Network'])
        self.log_params(**self.config['Preprocessing'])
        self.log_params(**self.config['Loss'])
        with open(os.path.join(self.network_savepoint, 'config.yaml'), 'w') as outfile:
            yaml.dump(self.config, outfile)
        self.log_artifact(os.path.join(self.network_savepoint, 'config.yaml'))

    def set_train_dataset(self, dataset):
        self.dataset = dataset

    def set_valid_dataset(self, dataset):
        self.validation_dataset = dataset

    def set_test_dataset(self, dataset):
        self.test_dataset = dataset

    def set_optimizer(self, **config):
        """
        :return: A partial function of an optimizers. Partial passed arguments are hyperparameters
        """
        solver = config.pop['solver']
        func = OPTIMS[solver]
        self.partial_optimizer = partial_fill_kwargs(func, config)

    def set_scheduler(self, **config):
        scheduler = config.pop['scheduler']
        scheduler_options = SCHEDULER[scheduler]
        self.ctx_train['scheduler_opt'] = {'call_on': scheduler[1]}
        self.ctx_train['scheduler_opt']['callback'] = scheduler[2]

        self.partial_lr_scheduler = partial_fill_kwargs(scheduler_options[0], config)

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
        class_count = self.dataset.get_class_count()
        kwargs = {}
        if 'params_weighting' in self.config['Training']:
            kwargs = self.config['Training']['params_weighting']

        return torch.tensor(class_weighting(class_count, ignore_index=self.ignore_index, **kwargs))

    def setup_class_weights(self, weights):
        self.class_weights = weights

    def save_model(self, model, filename, **kwargs):
        save = model.save(savepoint=self.network_savepoint, filename=filename, **kwargs)
        if 'best_valid' in filename:
            self.last_save['best_valid'] = save
        else:
            self.last_save['last'] = save
        if self.config['Manager']['max_saved_model']:
            files = glob.glob(self.network_savepoint + "/best_valid_*.pth")
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
                Tracker.warn("Attempt to register model at %s" % self.last_save)
                self.register_trained_model()
                run_end = (input("Call end function ? [y]/n ") or "y").lower()
                if run_end == 'y':
                    self.postpone_killed_flag = True
                else:
                    self.mlflow_client.set_terminated(self.run_id, status='KILLED')
            if self.multi_gpu:
                dist.barrier()
            if not self.postpone_killed_flag:
                self.clean_up()
                raise KeyboardInterrupt
        except:

            if self.is_main_process(rank):
                self.mlflow_client.set_terminated(self.run_id, status='FAILED')
            if self.multi_gpu:
                dist.barrier()
            self.clean_up()
            raise

        if self.is_main_process(rank) and self.run_training and not self.postpone_killed_flag:
            self.save_model(model, 'last')
            self.register_trained_model()
        if self.multi_gpu:
            dist.barrier()

        with autocast(enabled=self.config['Training']['amp']):
            self.end(model, rank)

        if self.postpone_killed_flag and self.is_main_process(rank):
            self.mlflow_client.set_terminated(self.run_id, status='KILLED')
        self.clean_up()

    def start(self):
        assert self.partial_optimizer is not None, "Missing optimizer for training"
        assert self.dataset is not None, "Missing dataset"

        self.postpone_killed_flag = False

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
        if self.register_params:
            self.initial_tracking()
        if self.multi_gpu:
            mp.spawn(self._start_process,
                     nprocs=self.world_size,
                     join=True)
        else:
            self._start_process(rank=0)

        if not self.postpone_killed_flag:
            self.mlflow_client.set_terminated(self.run_id, status='FINISHED')
        save_yaml(self.config, os.path.join(self.run_folder, 'config.yaml'))

    def register_trained_model(self):
        if self.last_save['best_valid']:
            self.log_artifact(self.last_save['best_valid'])
        if self.last_save['last']:
            self.log_artifact(self.last_save['last'])

    def end(self, model, rank):
        pass

    def train(self, model, rank=0):
        loss_function = self.get_loss(self.class_weights, rank=rank)
        optimizer = self.partial_optimizer(model.get_trainable_parameters(self.config['Optimizer']['params_solver']['lr']))
        if self.partial_lr_scheduler is not None:
            lr_scheduler = self.partial_lr_scheduler(optimizer)
        else:
            lr_scheduler = None
        iteration = self.current_iteration - 1
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
                iteration += 1
                img, gt = self.batch_to_device(batch, rank)
                pred = model(img)
                loss = loss_function(pred, gt) / iters_to_accumulate
                scaler.scale(loss).backward()
                if (i + 1) % iters_to_accumulate == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()

                if self.ctx_train['scheduler_opt']['call_on'] == 'on_iteration':
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

            if self.ctx_train['scheduler_opt']['call_on'] == 'on_epoch':
                self.lr_scheduler_step(lr_scheduler, e, iteration, len(train_loader))

            if self.is_main_process(rank):
                progressBar.close()

    def lr_scheduler_step(self, lr_scheduler, epoch, iteration, size_epoch, validation_metrics=None):
        if lr_scheduler is None:
            return

        if self.ctx_train['scheduler_opt']['call_on'] == 'on_validation':
            lr_scheduler.step(validation_metrics)
        else:
            lr_scheduler.step(self.ctx_train['scheduler_opt']['callback'](epoch, iteration, size_epoch))

    def validate(self, model, iteration, rank=0):
        pass
