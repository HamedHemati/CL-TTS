import os
import copy
import torch

from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, WandBLogger

from .base_trainer import BaseTrainer
from cl_tts.metrics import get_metrics
from cl_tts.strategies import get_strategy


class BaseCLTrainer(BaseTrainer):
    """
    Base class Trainer. All trainers should inherit from this class.
    """
    def __init__(self, args, params, experiment_name):
        super().__init__(args, params, experiment_name)

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
        metrics_list = get_metrics(metrics, self.params, self.benchmark_meta,
                                   self.log_to_wandb)

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
            self.model,
            self.optimizer,
            self.forward_func,
            self.criterion_func,
            self.benchmark_meta["collator"],
            self.evaluation_plugin,
            self.device
        )

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
        raise NotImplementedError()

    def generate_samples(self):
        raise NotImplementedError()
