import timm
from nntools.experiment.supervised_experiment import SupervisedExperiment
from torchmetrics import CohenKappa, JaccardIndex, F1Score, Accuracy


class ClassificationExperiment(SupervisedExperiment):
    def __init__(self, config, run_id=None, trial=None, multilabel=False):
        super().__init__(config, run_id, trial, multilabel)
        self.set_optimizer(**self.c['Optimizer'])

    def init_model(self):
        model_setup = self.c['Network'].copy()
        model_name = model_setup.pop('architecture')
        model_setup.pop('synchronized_batch_norm', None)
        n_classes = model_setup.pop('n_classes', None)
        model = timm.create_model(model_name, num_classes=n_classes, **model_setup)
        self.set_model(model)

        if self.multilabel:
            task = 'multilabel'
        else:
            task = 'multiclass'
        if n_classes <= 2:
            task = 'binary'

        self.model.add_metric({'Accuracy': Accuracy(num_classes=self.n_classes, task=task),
                               'F1': F1Score(num_classes=self.n_classes, task=task),
                               'Jaccard': JaccardIndex(num_classes=self.n_classes, task=task)})
        if task != 'multilabel':
            self.model.add_metric({'CohenKappa': CohenKappa(num_classes=self.n_classes, task=task)})
