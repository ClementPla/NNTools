import torch
import random
from torchmetrics import CohenKappa, JaccardIndex, Dice
from nntools.utils import torch
from nntools.experiment.supervised_experiment import SupervisedExperiment
import segmentation_models_pytorch as smp



class SegmentationExperiment(SupervisedExperiment):
    def __init__(self, config, run_id=None, trial=None, 
                 multilabel=False,
                 ignore_score_index=0):
        super().__init__(config, run_id, trial)
        
        
        self.gt_name = 'mask'
        self.data_keys = ['image']
        
        self.multilabel = multilabel
        self.ignore_score_index = ignore_score_index
        self.set_optimizer(**self.c['Optimizer'])

    
    def init_model(self):
        
        model_setup = self.c['Network'].copy()
        model_name = model_setup.pop('architecture')
        model_setup.pop('synchronized_batch_norm', None)
        n_classes = model_setup.pop('n_classes', None)
        model = smp.create_model(model_name, classes=n_classes, **model_setup)
        self.set_model(model)
        self.model.add_metric({'CohenKappa':CohenKappa(self.n_classes, weights='quadratic'),
                                'mIoU':JaccardIndex(self.n_classes, multilabel=self.multilabel, ignore_index=self.ignore_score_index),
                                'Dice':Dice(self.n_classes, ignore_index=self.ignore_score_index)})
    
    def validate(self, model, valid_loader, loss_function=None):
        index = random.randint(0, len(valid_loader))
        batch = valid_loader[index]
        with torch.no_grad():
            preds = model(batch['image'])
            if self.multilabel:
                preds = preds > 0.5
            else:
                preds = torch.argmax(preds, 1, keepdim=True)
        
        self.visualization_images(batch['image'], batch[self.gt_name], 'input_images')
        self.visualization_images(batch['image'], preds, 'output_images')        
        return super().validate(model, valid_loader, loss_function)