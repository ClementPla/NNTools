import numpy as np
import optuna
import torch
import torch.distributed as dist
import tqdm
from nntools import BINARY_MODE, MULTICLASS_MODE
from nntools.dataset import class_weighting
from nntools.nnet import FuseLoss, SUPPORTED_LOSS
from nntools.report.graph import build_bar_plot
from nntools.utils import reduce_tensor
from nntools.utils.misc import call_with_filtered_kwargs
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast
from .experiment import Experiment


class SupervisedExperiment(Experiment):
    def __init__(self, config, run_id=None, trial=None, multilabel=False):
        super(SupervisedExperiment, self).__init__(
            config, run_id=run_id, trial=trial)
        if 'ignore_index' in self.c['Training']:
            self.ignore_index = self.c['Training']['ignore_index']
        else:
            self.ignore_index = -100

        self.n_classes = config['Network'].get('n_classes', -1)
        self.class_weights = None

        self.multilabel = multilabel
        self.head_activation_exists = False
        self.n_validation_step = 0

    def datasets_summary(self):

        figTrain = build_bar_plot(self.train_dataset.get_class_count(load=True), 'Train dataset')
        self.tracker.log_figures([figTrain, 'train_data_count.png'])

        if self.validation_dataset:
            figValid = build_bar_plot(self.validation_dataset.get_class_count(load=True), 'Valid dataset')
            self.tracker.log_figures([figValid, 'valid_data_count.png'])

        for k, v in self.additional_datasets.items():
            d = v
            if not isinstance(d, list):
                d = [d]
                for i, dataset in enumerate(d):
                    if len(d) > 1:
                        subtitle = f'{i + 1}/{len(d) + 1}'
                    else:
                        subtitle = ''
                    figTest = build_bar_plot(dataset.get_class_count(load=True), f'{k} ' + subtitle)
                    self.tracker.log_figures(
                        [figTest, f'{k}_count_{subtitle}.png'])

        if self.test_dataset:
            d = self.test_dataset
            if not isinstance(d, list):
                d = [d]
            for i, dataset in enumerate(d):
                if len(d) > 1:
                    subtitle = f' {i + 1}/{len(d) + 1}'
                else:
                    subtitle = ''
                figTest = build_bar_plot(dataset.get_class_count(load=True), 'Test dataset ' + subtitle)
                self.tracker.log_figures(
                    [figTest, f'Test_data_count_{subtitle}.png'])

    def start(self, run_id=None):
        if self.c['Loss'].get('weighted_loss', False) and self.class_weights is None:
            class_weights = self.get_class_weights()
            self.setup_class_weights(weights=class_weights)
        if self.c['Manager'].get('log_dataset_summary', True):
            self.datasets_summary()
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
            self.class_weights = weights.float()

    def train(self, model, rank=0):
        loss_function = self.get_loss(self.class_weights, rank=rank)
        self.loss = loss_function
        super(SupervisedExperiment, self).train(model, rank)

    def in_epoch(self, model):
        clip_grad = self.c['Training'].get('grad_clipping', False)
        iters_to_accumulate = self.c['Training'].get('iters_to_accumulate', 1)
        moving_loss = []

        for i, batch in (enumerate(self.ctx.train_loader)):
            batch = self.batch_to_device(batch, rank=self.ctx.rank)
            will_validate = self.current_iteration % self.c['Validation']['log_interval'] == 0

            if ((i + 1) % iters_to_accumulate == 0) or will_validate:
                loss = self.forward_train(model, self.loss, batch)
                loss = loss / iters_to_accumulate
                self.ctx.scaler.scale(loss).backward()

                if clip_grad:
                    clip_grad_norm_(model.parameters(), float(clip_grad))
                self.ctx.scaler.step(self.ctx.optimizer)
                self.ctx.optimizer.zero_grad(set_to_none=True)
                self.ctx.scaler.update()
                model.zero_grad()
                self.update_scheduler_on_iteration()
            elif self.multi_gpu:
                with model.no_sync():
                    loss = self.forward_train(model, self.loss, batch)
                    loss = loss / iters_to_accumulate
                    self.ctx.scaler.scale(loss).backward()
            else:
                loss = self.forward_train(model, self.loss, batch)
                loss = loss / iters_to_accumulate
                self.ctx.scaler.scale(loss).backward()

            moving_loss.append(loss.detach().item())

            """
            Validation step
            """
            if will_validate:
                if moving_loss:
                    self.log_metrics(self.current_iteration,
                                     trainining_loss=np.mean(moving_loss))
                moving_loss = []
                self.save_model(model, filename='last')

                self.n_validation_step += 1
                self.in_validation(model)

            self.current_iteration += 1

            self.ctx.update_progress_bar()

        """ 
        If the validation set is not provided, we save the model once per epoch
        """
        if self.validation_dataset is None:
            self.save_model(model, filename='iteration_%i_loss_%f' % (self.current_iteration,
                                                                      float(np.mean(moving_loss))))
        self.update_scheduler_on_epoch()

    def in_validation(self, model):
        valid_loader = self.ctx.valid_loader
        if valid_loader is None:
            return

        with torch.no_grad():
            self.validate(model, valid_loader, self.loss)

        if self.multi_gpu:
            dist.barrier()

        current_metric = self.metrics[self.tracked_metric]
        if self.ctx.is_main_process:
            best_state_metric = self.get_state_metric()
            if (current_metric >= best_state_metric[self.tracked_metric]):
                filename = ('best_valid_iteration_%i_%s_%.3f' % (
                    self.current_iteration, self.tracked_metric, current_metric)).replace('.', '')
                self.save_model(model, filename=filename)

        if self._trial:
            if self.ctx.is_main_process:
                self._trial.report(current_metric, int(self.current_epoch))
            if self._trial.should_prune():
                if self.ctx.is_main_process:
                    self.tracker.log_metrics(self.current_iteration, pruned=1)
                    self.tracker.set_status('KILLED')
                    raise optuna.TrialPruned()

        self.update_scheduler_on_validation(-current_metric)
        model.train()

    def forward_train(self, model, loss_function, batch):
        with autocast(enabled=self.c['Manager']['amp']):
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

    def validate(self, model, valid_loader, loss_function=None):
        stats = self.eval_model(model, valid_loader,
                                loss_function=loss_function)
        self.log_metrics(step=self.current_iteration, **stats)

    def head_activation(self, preds):
        if self.head_activation_exists:
            return preds
        else:
            if self.multilabel:
                return torch.sigmoid(preds)
            else:
                return torch.softmax(preds, 1)

    def eval_model(self, model, dataloader, loss_function=None, log_loss=True):
        model._metrics.reset()
        losses = 0
        model.eval()
        with autocast(enabled=self.c['Manager']['amp']):
            with torch.no_grad():
                for n, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
                    batch = self.batch_to_device(batch, self.ctx.rank)
                    img = batch['image']
                    gt = batch[self.gt_name]
                    proba = model(img)
                    if loss_function:
                        losses += loss_function(proba, y_true=gt).detach()
                    proba = self.head_activation(proba)

                    if self.multilabel:
                        gt = gt.transpose(1, -1).flatten(0, -2)
                        proba = proba.transpose(1, -1).flatten(0, -2)
                    model._metrics.update(proba, gt)
                stats = {k: v for k, v in model._metrics.compute().items()}

                if loss_function:
                    if self.multi_gpu:
                        losses = reduce_tensor(
                            losses, self.world_size, mode='sum') / self.world_size
                    losses = losses / n
                    if log_loss:
                        stats['validation_loss'] = losses.item()

        model.train()
        return stats

    def end(self, model):
        rank = self.ctx.rank
        loss_function = self.get_loss(self.class_weights, rank=rank)
        self.loss = loss_function
        gpu = self.get_gpu_from_rank(rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load(self.tracker.network_savepoint, load_most_recent=True, map_location=map_location, strict=False,
                   filtername='best_valid')

        test_loader, test_sampler = self.get_dataloader(self.test_dataset, shuffle=False, batch_size=24,
                                                        rank=rank)
        stats = self.eval_model(model, test_loader, self.loss, log_loss=False)

        test_scores = {f'Test_{k}': v for k, v in stats.items()}

        self.log_metrics(step=0, **test_scores)
