import os
import yaml
import torch
from trainer import get_optimizer

from cl_tts.utils.generic import set_random_seed, update_config
from cl_tts.models import get_model, get_model_config
from cl_tts.benchmarks import get_benchmark


class BaseTrainer:
    """
    Base class Trainer. All trainers should inherit from this class.
    """
    def __init__(self, args, params, experiment_name):
        self.params = params

        # Seed
        set_random_seed(params["seed"])

        # Set compute device
        self.device = torch.device("cuda" if
                                   torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        # Params
        self.args = args
        self.params = params
        self.ds_path = os.path.join(params["datasets_root"],
                                    params["dataset_name"])
        self.experiment_name = experiment_name
        config = get_model_config(params, self.ds_path)

        # Initialize benchmark
        self.config = update_config(config, self.params)
        self.benchmark, self.benchmark_meta, self.config =\
            get_benchmark(params, self.ds_path, config)

        # Update config
        self.config = update_config(config, self.params)
        self.config.log_to_wandb = self.args.wandb_proj != ""

        # Get model
        self.model = get_model(
            params,
            self.config,
            self.benchmark_meta["ap"],
            self.benchmark_meta["tokenizer"],
            self.benchmark_meta["speaker_manager"],
        )

        # Optimizer
        self.optimizer = get_optimizer(
            optimizer_name=config.optimizer,
            optimizer_params=config.optimizer_params,
            lr=config.lr,
            model=self.model,
        )

        # Criterion
        self.criterion = self.model.get_criterion()

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
