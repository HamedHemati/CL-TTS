import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class GradClipper(SupervisedPlugin):
    """
    This metric will return a `float` value after
    mini-batch.
    """

    def before_update(
            self, strategy, *args, **kwargs
    ):
        """
        """
        if strategy.model.config.grad_clip > 0.0:
            _ = torch.nn.utils.clip_grad_norm_(
                strategy.model.parameters(),
                strategy.model.config.grad_clip
            )
