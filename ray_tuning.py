import argparse
import json
import os
from pathlib import Path

import torch

from ray import tune, init
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.utils.log import Verbosity

import ssnp

init(log_to_driver=False)


class HyperParameterTuning:
    MAX_EPOCHS = 300
    CPUS_AVAIL = 5
    GPUS_AVAIL = 0.25
    NUM_SAMPLES = 5

    seed = 42  # not used

    CONFIG = {
        "m": tune.grid_search([1, 5, 10, 50]),
        "M": tune.grid_search([1, 5, 10]),
        "samples": tune.grid_search([0.50, 0.75, 1.0]),
        "diffusion": tune.grid_search([True, False]),
    }


class ComGraphArguments:
    def __init__(self, dataset):
        self.model = 2
        if dataset in ["density", "component", "cut_ratio", "coreness"]:
            self.use_nodeid = False
            self.use_one = True
        else:
            self.use_nodeid = True
            self.use_one = False
        self.repeat = 1
        self.use_seed = False
        self.dataset = dataset
        self.use_deg = False
        self.use_maxzeroone = False
        self.stochastic = True
        self.views = 1
        self.use_gcn_conv = False


def ray_tune_helper(identifier, output_path, dataset):
    hyper_class = HyperParameterTuning

    scheduler = FIFOScheduler()
    scheduler.set_search_properties(metric='val_accuracy', mode='max')

    reporter = CLIReporter(metric_columns=["loss", "val_accuracy", "training_iteration", "test_accuracy"],
                           sort_by_metric=True, max_progress_rows=25, metric="val_accuracy", max_error_rows=0)
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
        resume="AUTO",
        raise_on_failed_trial=False,
        verbose=Verbosity.V1_EXPERIMENT,
    )
    best_trial = result.get_best_trial("val_accuracy", "max", "last")

    print("Best trial config: {}".format(best_trial))
    with open(f'{str(Path.home())}/{identifier}_best_result.json', "w") as file:
        json.dump(best_trial.config, file)

    print("Best trial final train loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["val_accuracy"]))

    print("Final Table")
    reporter._max_progress_rows = 4 * 3 * 3 * 2 * 1
    reporter.report(result.trials, done=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    ray_tune_helper(args.identifier, args.output_path, args.dataset)
