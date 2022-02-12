import os
import yaml
import torch
import copy

import torch
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, WandBLogger

from cl_tts.utils.generic import set_random_seed
from cl_tts.models import get_model
from cl_tts.benchmarks import get_benchmark
from cl_tts.metrics import get_metrics
from cl_tts.strategies import get_strategy


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

        # Get model
        n_symbols = self.benchmark_meta["n_symbols"]
        n_speakers = self.benchmark_meta["n_speakers"]
        model, forward_func, criterion_func = get_model(params,
                                                        n_symbols,
                                                        n_speakers,
                                                        self.device)

        # Optimizer
        if params["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(),
                                         **params["optimizer_params"])
        else:
            raise NotImplementedError

        # ====================================
        #          Evaluation Plugin
        # ====================================
        # Initialize loggers
        loggers = [InteractiveLogger()]
        self.log_to_wandb = args.wandb_proj != ""
        if self.log_to_wandb:
            config = copy.copy(params)
            config.update(vars(args))
            wandb_logger = WandBLogger(
                project_name=args.wandb_proj,
                run_name=self.experiment_name,
                config=config,
            )
            loggers.append(wandb_logger)

        # Initialize metrics
        metrics = params["metrics"]
        metrics_list = get_metrics(metrics)

        # Set evaluation plugin
        self.evaluation_plugin = EvaluationPlugin(
            *metrics_list,
            loggers=loggers,
            benchmark=self.benchmark
        )

        # ====================================
        #              Strategy
        # ====================================
        self.strategy = get_strategy(
            self.params,
            model,
            optimizer,
            forward_func,
            criterion_func,
            self.benchmark_meta["collator"],
            self.evaluation_plugin,
            self.device
        )

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

    def _save_checkpoint(self, speaker, itr):
        checkpoint_path = os.path.join(self.checkpoints_path,
                                       f"best_{itr}_{speaker}.pt")
        torch.save(self.strategy.model.state_dict(), checkpoint_path)

    def _load_checkpoint(self):
        # Load checkpoint
        print(f"Loading checkpoint from  " + \
              f"{self.params['load_checkpoint_path']}")
        ckpt = torch.load(self.params["load_checkpoint_path"],
                          map_location=self.device)
        for name, param in self.strategy.model.named_parameters():
            try:
                self.strategy.model.state_dict()[name].copy_(ckpt[name])
            except:
                print(f"Could not load weights for {name}")

    def run(self):
        pass

    def generate_samples(self):
        pass
