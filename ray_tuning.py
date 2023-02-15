import argparse
import json
import os
import time
from pathlib import Path

import torch
from ray import tune
from ray.tune import CLIReporter, Stopper
from ray.tune.schedulers import ASHAScheduler

import GLASSTest


class TimeStopper(Stopper):
    def __init__(self):
        self._start = time.time()
        self._deadline = 60 * 45  # 45 minutes max run across all experiments

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time.time() - self._start > self._deadline


class HyperParameterTuning:
    MAX_EPOCHS = 300
    CPUS_AVAIL = 20
    GPUS_AVAIL = 1
    NUM_SAMPLES = 1

    seed = 42

    CONFIG = {
        "m": tune.grid_search([1, 5, 10, 25, 50]),
        "M": tune.grid_search([1, 5, 10, 25, 50]),
        "samples": tune.grid_search([0.25, 0.50, 0.75, 1.0]),
        "diffusion": tune.grid_search([True, False]),
    }


class ComGraphArguments:
    def __init__(self, dataset):
        self.model = 2
        self.use_nodeid = True
        self.repeat = 1
        self.use_seed = False
        self.dataset = dataset
        self.use_deg = False
        self.use_one = False
        self.use_maxzeroone = False
        self.stochastic = True


def ray_tune_helper(identifier, output_path, dataset):
    hyper_class = HyperParameterTuning

    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=hyper_class.MAX_EPOCHS,
        grace_period=32,
        reduction_factor=4)

    reporter = CLIReporter(metric_columns=["loss", "val_accuracy", "training_iteration"])
    base_args = ComGraphArguments(dataset)

    device = torch.device('cuda')
    print(f"Using device: {device} for running ray tune")

    result = tune.run(
        tune.with_parameters(GLASSTest.ray_tune_run_helper, argument_class=base_args, device=0),
        resources_per_trial={"cpu": hyper_class.CPUS_AVAIL, "gpu": hyper_class.GPUS_AVAIL},
        config=hyper_class.CONFIG,
        num_samples=hyper_class.NUM_SAMPLES,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.join(identifier, output_path),
        log_to_file=True,
        stop=TimeStopper(),
        resume="AUTO",
        raise_on_failed_trial=False
    )
    best_trial = result.get_best_trial("val_accuracy", "max", "last")

    print("Best trial config: {}".format(best_trial))
    with open(f'{str(Path.home())}/{identifier}_best_result.json', "w") as file:
        json.dump(best_trial.config, file)

    print("Best trial final train loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["val_accuracy"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    ray_tune_helper(args.identifier, args.output_path, args.dataset)
