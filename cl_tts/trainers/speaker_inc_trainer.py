from .base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, args, params, experiment_name):
        super().__init__(args, params, experiment_name)

    def run(self):
        for exp in self.benchmark.train_stream:
            self.strategy.train(exp)
