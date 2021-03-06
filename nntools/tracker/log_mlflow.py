import os
import time


def log_params(tracker, **params):
    run_id = tracker.run_id
    client = tracker.client
    for k, v in params.items():
        client.log_param(run_id, k, v)


def log_metrics(tracker, step, **metrics):
    run_id = tracker.run_id
    client = tracker.client
    for k, v in metrics.items():
        client.log_metric(run_id, k, v, int(time.time() * 1000), step=step)


def set_tags(tracker, **tags):
    run_id = tracker.run_id
    client = tracker.client
    for k, v in tags.items():
        client.set_tag(run_id, k, v)


def log_artifact(tracker, *paths):
    run_id = tracker.run_id
    client = tracker.client
    for p in paths:
        b = os.path.getsize(p)
        if b > 10e6:
            print("File %s won't be stored as an artifact (oversize limit)" % p)
        else:
            client.log_artifact(run_id, p)
