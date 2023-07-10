import os
from typing import Optional

from mlflow.tracking.client import MlflowClient

from nntools.utils.io import create_folder

from .log_mlflow import log_artifact, log_figures, log_metrics, log_params, set_tags


class Tracker:
    def __init__(self, exp_name, run_id=None, tracker_uri=None):
        self.exp_name = exp_name
        self.run_id = run_id
        self.save_paths = {}
        self.current_iteration = 0
        self._metrics = []
        self._params = []
        self._tags = []
        self._artifacts = []
        self._figures = []

        self.run_started = False
        self.register_params = True
        self.client = None
        self.exp_id = None
        if tracker_uri:
            self.create_client(tracker_uri)

    def register_path(self, key: str, path: str):
        self.save_paths[key] = path
        create_folder(path)
        self.__dict__.update(self.save_paths)

    def set_run_folder(self, path: str):
        self.register_path("run_folder", path)

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
        if self.register_params:
            if self.run_started:
                log_params(self, **params)
            else:
                self._params.append(params)

    def log_artifacts(self, *paths):
        if self.run_started:
            log_artifact(self, *paths)
        else:
            self._artifacts.append(paths)

    def log_figures(self, *figures):
        if self.run_started:
            log_figures(self, *figures)

        else:
            self._figures.append(figures)

    def set_tags(self, **tags):
        if self.register_params:
            if self.run_started:
                set_tags(self, **tags)
            else:
                self._tags.append(tags)

    def list_existing_runs(self):
        return self.client.search_runs(self.exp_id)

    def check_is_run_exists(self, run_id):
        list_runs = self.list_existing_runs()
        return any([r.info.run_id == run_id for r in list_runs])

    def check_run_status(self, run_id):
        if not self.check_is_run_exists(run_id=run_id):
            return "ABSENT"
        return self.get_run(run_id).info.status

    def create_run(self, tags=None, run_name=None):
        if tags is None:
            tags = {}
        run = self.client.create_run(experiment_id=self.exp_id, tags=tags, run_name=run_name)
        self.run_id = run.info.run_id
        self.run_started = True
        self.initialize_run()

    def initialize_run(self):
        if not self.run_started:
            self.run_started = True

        for step, metrics in self._metrics:
            self.log_metrics(step, **metrics)
        for params in self._params:
            self.log_params(**params)
        for tags in self._tags:
            self.set_tags(**tags)
        for p in self._artifacts:
            self.log_artifacts(*p)
        for f in self._figures:
            self.log_figures(*f)
        return True

    def get_run(self, run_id: str = Optional[None]):
        if run_id is None:
            run_id = self.run_id
        return self.client.get_run(run_id)

    def set_run_id(self, run_id: str):
        self.run_id = run_id

    def set_status(self, status: str):
        self.client.set_terminated(self.run_id, status)

    def go_to_exp_last_iteration(self):
        run = self.get_run()
        for k, v in run.data.metrics.items():
            his = self.client.get_metric_history(self.run_id, k)
            self.current_iteration = max(self.current_iteration, his[-1].step)

    def init_default_path(self):
        assert "run_folder" in self.save_paths

        self.register_path("network_savepoint", self.create_new_folder_in_run("trained_model"))
        self.register_path("prediction_savepoint", self.create_new_folder_in_run("predictions"))
        self.register_path("config_savepoint", self.create_new_folder_in_run("config"))
        self.register_path("validation", self.create_new_folder_in_run("validation"))

    def create_new_folder_in_run(self, folder):
        path = os.path.join(self.run_folder, folder, str(self.run_id))
        create_folder(path)
        return path

    def get_metric_history(self, metric: str):
        return self.client.get_metric_history(self.run_id, metric)

    def get_best_score_for_metric(self, metric: str, maximize=True):
        metric_history = self.get_metric_history(metric)
        if maximize:
            return max([m.value for m in metric_history])
        else:
            return min([m.value for m in metric_history])

    def last_metrics(self):
        return self.client.get_run(self.run_id).data.metrics
