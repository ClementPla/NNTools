import torch
from torchmetrics import CohenKappa, JaccardIndex, F1Score, Accuracy
from nntools.experiment.supervised_experiment import SupervisedExperiment
import timm



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
        
        self.model.add_metric({'CohenKappa':CohenKappa(self.n_classes),
                                'Accuracy':Accuracy(num_classes=self.n_classes),
                                'F1':F1Score(num_classes=self.n_classes),
                                'Jaccard':JaccardIndex(num_classes=self.n_classes)})