import logging
from typing import List

from overrides import overrides
import torch

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import LearningRateScheduler
from dialogue_models.training.util import annealing_cos
import numpy as np


logger = logging.getLogger(__name__)


@LearningRateScheduler.register("one_cycle")
class OneCycle(LearningRateScheduler):
    """

    Parameters
    ----------
    num_epochs : ``int``, required.
        The total number of epochs for which the model should be trained.
    num_steps_per_epoch: ``int``, required.
        The number of steps (updates, batches) per training epoch.
    cut_frac: ``float``, optional (default = 0.1).
        The fraction of the steps to increase the learning rate.
    ratio: ``float``, optional (default = 32).
        The ratio of the smallest to the (largest) base learning rate.
    gradual_unfreezing: ``bool``, optional (default = False).
        Whether gradual unfreezing should be used.
    discriminative_fine_tuning: ``bool``, optional (default = False).
        Whether discriminative fine-tuning (different learning rates per layer)
        are used.
    decay_factor: ``float``, optional (default = 0.38).
        The decay factor by which the learning rate is reduced with
        discriminative fine-tuning when going a layer deeper.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 num_epochs: int,
                 num_steps_per_epoch: int,
                 cut_frac: float = 0.25,
                 ratio: int = 25,
                 last_epoch: int = -1,
                 final_div: float = None,
                 decay_factor: float = 0.38) -> None:


        self.final_div = final_div
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.is_first_epoch = True
        self.batch_num_total_epoch_end: List[int] = []
        super().__init__(optimizer, last_epoch)
        # track the actual number of steps for each epoch
        # set up for the first batch
        self.last_batch_num_total = -1
        self.step_batch(0)

        if self.final_div is None: self.final_div = ratio*1e4

    @overrides
    def step(self, metric: float = None, epoch: int = None) -> None:
        if len(self.batch_num_total_epoch_end) == 0: # pylint: disable=len-as-condition
            self.batch_num_total_epoch_end.append(0)
        else:
            self.batch_num_total_epoch_end.append(self.last_batch_num_total)

    def step_batch(self, batch_num_total: int = None):
        if batch_num_total is None:
            batch_num_total = self.last_batch_num_total + 1
        self.last_batch_num_total = batch_num_total
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_values()):
            param_group['lr'] = learning_rate

    def get_values(self):
        # get the actual number of batches per epoch seen in training
        if len(self.batch_num_total_epoch_end) > 1:
            # have finished an epoch
            actual_num_steps_per_epoch = int(
                    self.batch_num_total_epoch_end[-1] /
                    (len(self.batch_num_total_epoch_end) - 1)
            )
        else:
            actual_num_steps_per_epoch = max(self.num_steps_per_epoch,
                                             self.last_batch_num_total)

        # otherwise we use the schedule for the rest of training
        num_steps = self.num_epochs * actual_num_steps_per_epoch
        step = min(self.last_batch_num_total, num_steps)

        cut = int(num_steps * self.cut_frac)

        if step < cut:
            prop = step / cut
            result = [annealing_cos(lr / self.ratio, lr, prop) for lr in self.base_values]
            return result
        else:
            prop = (step - cut) / (num_steps - cut)
            return [annealing_cos(lr, lr /self.final_div, prop) for lr in self.base_values]
