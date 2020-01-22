
import torch
from allennlp.training.momentum_schedulers.momentum_scheduler import MomentumScheduler
from dialogue_models.training.optimizers import AdamW
from dialogue_models.training.util import annealing_cos


@MomentumScheduler.register("one_cycle")
class OneCycleMomentumScheduler(MomentumScheduler):
    """
    Adjust momentum during training according to an inverted triangle-like schedule.

    The momentum starts off high, then decreases linearly for ``cool_down`` epochs,
    until reaching ``1 / ratio`` th of the original value. Then the momentum increases
    linearly for ``warm_up`` epochs until reaching its original value again. If there
    are still more epochs left over to train, the momentum will stay flat at the original
    value.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cool_down: int,
                 warm_up: int,
                 num_steps_per_epoch: int,
                 ratio: int = 10,
                 last_epoch: int = -1) -> None:
        self.cool_down = cool_down
        self.warm_up = warm_up
        self.num_steps_per_epoch = num_steps_per_epoch
        self.ratio = ratio
        self.last_batch_num_total = -1
        super().__init__(optimizer, last_epoch)
        self.step_batch(0)

    def step_batch(self, batch_num_total: int = None):
        if batch_num_total is None:
            batch_num_total = self.last_batch_num_total + 1
        self.last_batch_num_total = batch_num_total
        for param_group, momentum in zip(self.optimizer.param_groups, self.get_values()):
            param_group[self.param_group_field] = momentum


    def get_values(self):
        # step = self.last_epoch + 1
        step = self.last_batch_num_total / self.num_steps_per_epoch
        base_values = self.base_values

        # unpack Adam params
        if not isinstance(self.optimizer, torch.optim.SGD):
            base_values, b2 = zip(*self.base_values)

        if step <= self.cool_down:

            prop = step / self.cool_down
            values = [annealing_cos(m, m / self.ratio, prop) for m in base_values]


            # values = [m  - (m - m / self.ratio) * (step / self.cool_down)
            #           for m in base_values]
        elif step <= self.cool_down + self.warm_up:
            prop = (step - self.cool_down) / (self.warm_up)
            values = [annealing_cos(m / self.ratio, m, prop) for m in base_values]

            # values = [(m / self.ratio) + (m - m / self.ratio) * (step - self.cool_down) / self.warm_up
            #           for m in base_values]
        else:
            values = base_values

        # repack adam params
        if not isinstance(self.optimizer, torch.optim.SGD):
            values = list(zip(values, b2))

        return values

    # def get_values(self):
    #     if isinstance(self.optimizer, torch.optim.SGD):
    #         step = self.last_epoch + 1
    #         if step <= self.cool_down:
    #             values = [m  - (m - m / self.ratio) * (step / self.cool_down)
    #                       for m in self.base_values]
    #         elif step <= self.cool_down + self.warm_up:
    #             values = [(m / self.ratio) + (m - m / self.ratio) * (step - self.cool_down) / self.warm_up
    #                       for m in self.base_values]
    #         else:
    #             values = self.base_values
    #
    #     elif isinstance(self.optimizer, (torch.optim.Adam, AdamW)):
    #
    #         step = self.last_epoch + 1
    #         if step <= self.cool_down:
    #             values = [(b1  - (b1 - b1 / self.ratio) * (step / self.cool_down), b2)
    #                       for b1, b2 in self.base_values]
    #         elif step <= self.cool_down + self.warm_up:
    #             values = [((b1 / self.ratio) + (b1 - b1 / self.ratio) * (step - self.cool_down) / self.warm_up, b2)
    #                       for b1, b2 in self.base_values]
    #         else:
    #             values = self.base_values
    #     else:
    #         raise ValueError()
    #
    #     return values
