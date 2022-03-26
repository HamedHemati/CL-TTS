import os
import yaml
import torch

from cl_tts.utils.generic import set_random_seed
from cl_tts.models import get_model, get_model_config
from cl_tts.benchmarks import get_benchmark


class BaseTrainer:
    """
    Base class Trainer. All trainers should inherit from this class.
    """
    def __init__(self, args, params, experiment_name):
        self.params = params

        set_random_seed(params["seed"])
        self.args = args
        self.params = params
        self.experiment_name = experiment_name

        # Set compute device
        self.device = torch.device("cuda" if
                                   torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        self.ds_path = os.path.join(params["datasets_root"],
                                    params["dataset_name"])

        # Config
        config = get_model_config(params, self.ds_path)

        # Initialize benchmark
        self.benchmark, self.benchmark_meta, self.config =\
            get_benchmark(params, self.ds_path, config)

        # Get model
        self.model = get_model(
            params,
            self.config,
            self.benchmark_meta["ap"],
            self.benchmark_meta["tokenizer"],
            self.benchmark_meta["speaker_manager"],
        )

        # TODO: Forward and Backward functions

        # Optimizer
        if params["optimizer"] == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              **params["optimizer_params"])
        else:
            raise NotImplementedError

        # Results
        if self.args.save_results:
            # Results path
            results_path = os.path.join(self.args.outputs_dir,
                                        "outputs", self.experiment_name)
            self.results_path = results_path
            os.makedirs(results_path, exist_ok=True)

            # Checkpoints path
            self.checkpoints_path = os.path.join(self.results_path,
                                                 "checkpoints")
            os.makedirs(self.checkpoints_path, exist_ok=True)

            # Save a copy of params to the results folder
            output_params_yml_path = os.path.join(self.results_path,
                                                  "params.yml")
            with open(output_params_yml_path, 'w') as outfile:
                yaml.dump(self.params, outfile, default_flow_style=False)

    def run(self):
        raise NotImplementedError()

    def generate_samples(self):
        raise NotImplementedError()
