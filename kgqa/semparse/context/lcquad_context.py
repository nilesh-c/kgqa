from typing import List
from allennlp.data import Token

from kgqa.semparse.executor.executor import Executor


class LCQuADContext:
    executor = None

    def __init__(self, executor: Executor,
                 question_tokens: List[Token],
                 question_entities: List[str],
                 question_predicates: List[str]):
        self.executor = executor
        self.question_tokens = question_tokens
        self.question_entities = question_entities
        self.question_predicates = question_predicates
