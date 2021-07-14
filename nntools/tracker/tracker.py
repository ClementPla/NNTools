import os

from mlflow.tracking.client import MlflowClient
from nntools.utils.io import create_folder

from .log_mlflow import log_metrics, log_params, log_artifact, set_tags


class Tracker:
    def __init__(self, exp_name, run_id=None):
        self.exp_name = exp_name
        self.run_id = run_id
        self.save_paths = {}
        self.current_iteration = 0
        self._metrics = []
        self._params = []
        self._tags = []
        self._artifacts = []
        self.run_started = False

    def add_path(self, key, path):
        self.save_paths[key] = path
        create_folder(path)
        self.__dict__.update(self.save_paths)

    def set_run_folder(self, path):
        self.add_path('run_folder', path)

    def create_client(self, tracker_uri, artifact_uri=None):
        self.client = MlflowClient(tracker_uri)
        exp = self.client.get_experiment_by_name(self.exp_name)
        if exp is None:
            self.exp_id = self.client.create_experiment(self.exp_name, artifact_location=artifact_uri)
        else:
            self.exp_id = exp.experiment_id

    def log_metrics(self, step, **metrics):
        if self.run_started:
            log_metrics(self, step, **metrics)
        else:
            self._metrics.append((step, metrics))

    def log_params(self, **params):
        if self.run_started:
            log_params(self, **params)
        else:
            self._params.append(params)

    def log_artifacts(self, *paths):
        if self.run_started:
            log_artifact(self, *paths)
        else:
            self._artifacts.append(paths)

    def set_tags(self, **tags):
        if self.run_started:
            set_tags(self, **tags)
        else:
            self._tags.append(tags)

    def create_run(self, tags=None):
        if tags is None:
            tags = {}
        run = self.client.create_run(experiment_id=self.exp_id, tags=tags)
        self.run_id = run.info.run_id
        self.run_started = True
        self.initialize_run()

    def initialize_run(self):
        if not self.run_started:
            return False
        for step, metrics in self._metrics:
            self.log_metrics(step, **metrics)
        for params in self._params:
            self.log_params(**params)
        for tags in self._tags:
            self.set_tags(**tags)
        for p in self._artifacts:
            self.log_artifacts(*p)
        return True

    def get_run(self, id=None):
        if id is not None:
            self.run_id = id
        self.run_started = True
        self.initialize_run()
        return self.client.get_run(self.run_id)

    def set_status(self, status):
        self.client.set_terminated(self.run_id, status)

    def go_to_exp_last_iteration(self):
        run = self.get_run()
        for k, v in run.data.metrics.items():
            his = self.client.get_metric_history(self.run_id, k)
            self.current_iteration = max(self.current_iteration, his[-1].step)

    def init_default_path(self):
        assert 'run_folder' in self.save_paths
        self.add_path('network_savepoint', os.path.join(self.run_folder, 'trained_model', str(self.run_id)))
        self.add_path('prediction_savepoint', os.path.join(self.run_folder, 'predictions', str(self.run_id)))
