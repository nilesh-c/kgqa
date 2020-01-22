# import torch
#
# from allennlp.common.params import Params
# from allennlp.common.registrable import Registrable
# from allennlp.training.scheduler import Scheduler
#
# from dialogue_models.training.optimizers import AdamW
#
#
# class MomentumScheduler(Scheduler, Registrable):
#
#     def __init__(self,
#                  optimizer: torch.optim.Optimizer,
#                  last_epoch: int = -1) -> None:
#
#         if isinstance(optimizer, torch.optim.SGD):
#             param_group_field = "momentum"
#         elif isinstance(optimizer, (torch.optim.Adam, AdamW)):
#             param_group_field = "betas"
#         else:
#             raise ValueError()
#
#         super().__init__(optimizer, param_group_field, last_epoch)
#
#     def get_values(self) -> None:
#         raise NotImplementedError
#
#     # Requires custom from_params so we can pass the optimizer.
#     @classmethod
#     def from_params(cls, optimizer: torch.optim.Optimizer, params: Params):  # type: ignore
#         # pylint: disable=arguments-differ
#         scheduler_type = params.pop_choice("type", MomentumScheduler.list_available())
#         scheduler = MomentumScheduler.by_name(scheduler_type)(optimizer, **params.as_dict())
#         return scheduler