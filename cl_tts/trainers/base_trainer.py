import os
import yaml
import torch

from cl_tts.utils.generic import set_random_seed
from cl_tts.models import get_model
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

        # Initialize benchmark
        self.benchmark, self.benchmark_meta =\
            get_benchmark(self.args, self.params)
        self.benchmark_meta["ap_params"] = params["ap_params"]

        # Get model
        n_symbols = self.benchmark_meta["n_symbols"]
        n_speakers = self.benchmark_meta["n_speakers_dataset"]
        self.model, self.forward_func, self.criterion_func = get_model(
            params,
            n_symbols,
            n_speakers,
            self.device
        )

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
