from typing import Dict, Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator

from .base_strategy import BaseStrategy


class Naive(BaseStrategy):
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            params: Dict,
            forward_func: Callable,
            criterion_func: Callable,
            *,
            num_workers: int = 4,
            device="cpu",
            plugins=None,
            evaluator: EvaluationPlugin = default_evaluator,
    ):
        super(Naive, self).__init__(
            model=model,
            optimizer=optimizer,
            params=params,
            forward_func=forward_func,
            criterion_func=criterion_func,
            num_workers=num_workers,
            device=device,
            plugins=plugins,
            evaluator=evaluator
        )

    def training_epoch(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
