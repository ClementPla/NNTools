import optuna
from optuna.trial import Trial

class PrunningHandler(object):
    def __init__(self, trial:Trial, metric:str, trainer) -> None:
        self._trial = trial
        self._metric = metric
        self._trainer = trainer
    
    def __call__(self, engine: "Engine") -> None:
        score = self._trainer.metrics[self._metric]
        epoch = self._trainer.current_epoch
        self._trial.report(score, epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at {} epoch.".format(self._trainer.state.epoch)
            raise optuna.TrialPruned(message)
        
        
