from overrides import overrides
from typing import List

from allennlp.training.metrics.metric import Metric


@Metric.register("average_recall")
class AverageRecall(Metric):
    """
    This :class:`Metric` stores the average recall.
    """
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, relevant: List[str], retrieved: List[str]):
        """
        Adds instant's recall to the total recall.

        Parameters
        ----------
        relevant : ``List[str]``
            List of relevant answers.
        retrieved : ``List[str]``
            List of retrieved answers.
        """
        relevant, retrieved = set(relevant), set(retrieved)
        if len(relevant) != 0:
            instance_recall = len(relevant.intersection(retrieved)) / len(relevant)
        else:
            raise ValueError
        self._total_value += instance_recall
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Get the average recall.

        Parameters
        ----------
        reset : ``Bool`` (optional, default=False)
            Whether to reset the average recall after returning its current value.

        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        """
        Resets the average recall to zero.
        """
        self._total_value = 0.0
        self._count = 0

