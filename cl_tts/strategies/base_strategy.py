from typing import Dict, Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.templates.base_sgd import BaseSGDTemplate
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator


class BaseStrategy(BaseSGDTemplate):
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            params: Dict,
            forward_func: Callable,
            criterion_func: Callable,
            *,
            device="cpu",
            plugins=None,
            evaluator: EvaluationPlugin = default_evaluator,
    ):
        """

        :param model:
        :param optimizer:
        :param params:
        :param forward_func:
        :param criterion_func:
        :param device:
        :param plugins:
        :param evaluator:
        """
        super().__init__(model, optimizer,
                         train_mb_size=params["train_mb_size"],
                         train_epochs=params["train_epochs"],
                         eval_mb_size=params["eval_mb_size"],
                         device=device, plugins=plugins, evaluator=evaluator,
                         eval_every=params["eval_every"],
                         peval_mode=params["peval_mode"])

        self.forward_func = forward_func
        self.criterion_func = criterion_func

    def training_epoch(self, **kwargs):
        raise NotImplementedError

    def make_train_dataloader(self, **kwargs):
        pass

    def make_eval_dataloader(self, **kwargs):
        pass

    def forward(self):
        """ Forward function for `self.mbatch` """
        return self.forward_func(self.mbatch)

    def criterion(self):
        """ Criterion function for self.mb_out """
        return self.criterion_func(self.mb_output, self.mbatch)
