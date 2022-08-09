import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

import nntools.tracker.metrics as NNmetrics
from nntools import BINARY_MODE, MULTICLASS_MODE
from nntools.dataset import class_weighting
from nntools.nnet import FuseLoss, SUPPORTED_LOSS
from nntools.utils import reduce_tensor
from nntools.utils.misc import call_with_filtered_kwargs
from .experiment import Experiment
import optuna


class SupervisedExperiment(Experiment):
    def __init__(self, config, run_id=None, tracked_metric='mIoU', trial=None):
        super(SupervisedExperiment, self).__init__(
            config, run_id=run_id, tracked_metric=tracked_metric, trial=trial)
        if 'ignore_index' in self.c['Training']:
            self.ignore_index = self.c['Training']['ignore_index']
        else:
            self.ignore_index = -100

        self.n_classes = config['Network'].get('n_classes', -1)
        self.class_weights = None
        self.gt_name = 'mask'
        self.data_keys = ['image']

    def start(self, run_id=None):
        if self.c['Loss'].get('weighted_loss', False) and self.class_weights is None:
            class_weights = self.get_class_weights()
            self.setup_class_weights(weights=class_weights)
        super(SupervisedExperiment, self).start(run_id)

    def get_loss(self, weights: torch.Tensor = None, rank=0) -> FuseLoss:
        config = self.c['Loss']
        mode = MULTICLASS_MODE if self.n_classes > 2 else BINARY_MODE
        fuse_loss = FuseLoss(fusion=config.get('fusion', 'mean'), mode=mode)
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

    def get_class_weights(self) -> torch.Tensor:
        class_count = self.train_dataset.get_class_count()
        kwargs = self.c['Loss'].get('params_weighting', {})
        return torch.tensor(class_weighting(class_count, ignore_index=self.ignore_index, **kwargs))

    def setup_class_weights(self, weights: torch.Tensor):
        if self.c['Manager']['amp']:
            self.class_weights = weights.half()
        else:
            self.class_weights = weights

    def train(self, model, rank=0):
        loss_function = self.get_loss(self.class_weights, rank=rank)
        self.loss = loss_function
        super(SupervisedExperiment, self).train(model, rank)

    def in_epoch(self, model):
        clip_grad = self.c['Training'].get('grad_clipping', False)
        iters_to_accumulate = self.c['Training'].get('iters_to_accumulate', 1)
        moving_loss = []

        for i, batch in (enumerate(self.ctx.train_loader)):
            self.current_iteration += 1
            with autocast(enabled=self.c['Manager']['amp']):
                batch = self.batch_to_device(batch, rank=self.ctx.rank)
                loss = self.forward_train(self.model, self.loss, batch)
                loss = loss / iters_to_accumulate
                self.ctx.scaler.scale(loss).backward()
                if (i + 1) % iters_to_accumulate == 0:
                    if clip_grad:
                        clip_grad_norm_(model.parameters(), float(clip_grad))
                    self.ctx.scaler.step(self.ctx.optimizer)
                    self.ctx.scaler.update()
                    model.zero_grad()
                    self.update_scheduler_on_iteration()
                moving_loss.append(loss.detach().item())

            """
            Validation step
            """
            if self.current_iteration % self.c['Validation']['log_interval'] == 0:
                if moving_loss:
                    self.log_metrics(self.current_iteration,
                                     trainining_loss=np.mean(moving_loss))
                moving_loss = []
                self.save_model(model, filename='last')
                self.in_validation()

            self.ctx.update_progress_bar()

        """ 
        If the validation set is not provided, we save the model once per epoch
        """
        if self.validation_dataset is None:
            self.save_model(model, filename='iteration_%i_loss_%f' % (self.current_iteration,
                                                                      float(np.mean(moving_loss))))
        self.update_scheduler_on_epoch()

    def in_validation(self):
        model = self.ctx.model
        valid_loader = self.ctx.valid_loader

        if valid_loader is not None:
            with torch.no_grad():
                with autocast(enabled=self.c['Manager'].get('amp', False)):
                    valid_metric = self.validate(model, valid_loader,
                                                 self.current_iteration,
                                                 self.loss)

            if self._trial:
                self._trial.report(valid_metric, self.current_iteration)
                if self._trial.should_prune():
                    raise optuna.TrialPruned()
            self.update_scheduler_on_validation(valid_metric)

    def forward_train(self, model, loss_function, batch):
        pred = model(*self.pass_data_keys_to_model(batch))
        if isinstance(pred, tuple):
            loss = loss_function(*pred, y_true=batch[self.gt_name])
        else:
            loss = loss_function(pred, y_true=batch[self.gt_name])
        return loss

    def pass_data_keys_to_model(self, batch):
        args = []
        for key in self.data_keys:
            if key in batch:
                args.append(batch[key])
        return tuple(args)

    def validate(self, model, valid_loader, iteration, loss_function=None):
        model._metrics.reset()
        gpu = self.get_gpu_from_rank(self.ctx.rank)
        confMat = torch.zeros(self.n_classes, self.n_classes).cuda(gpu)
        losses = 0
        model.eval()
        for n, batch in enumerate(valid_loader):
            batch = self.batch_to_device(batch, self.ctx.rank)
            img = batch['image']
            gt = batch[self.gt_name]
            proba = model(img)
            losses += loss_function(proba, y_true=gt).detach()
            model._metrics.update(proba, gt)
            pred = torch.argmax(proba, 1)
            confMat += NNmetrics.confusion_matrix(pred,
                                                  gt, num_classes=self.n_classes)
        if self.multi_gpu:
            confMat = reduce_tensor(confMat, self.world_size, mode='sum')
            losses = reduce_tensor(
                losses, self.world_size, mode='sum') / self.world_size
        losses = losses / n
        confMat = NNmetrics.filter_index_cm(confMat, self.ignore_index)
        mIoU = NNmetrics.mIoU_cm(confMat)
        stats = NNmetrics.report_cm(confMat)
        stats['mIoU'] = mIoU
        stats['validation_loss'] = losses.item()
        stats.update(model._metrics.compute())
        self.log_metrics(step=iteration, **stats)

        best_state_metric = self.get_state_metric()
        current_metric = self.metrics[self.tracked_metric]
        if(current_metric >= best_state_metric[self.tracked_metric]):
            filename = ('best_valid_iteration_%i_%s_%.3f' % (
                iteration, self.tracked_metric,  current_metric)).replace('.', '')
            self.save_model(model, filename=filename)
        model.train()

        return current_metric
