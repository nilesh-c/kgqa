
from pathlib import Path
from typing import *
import torch
import torch.optim as optim
import numpy as np
import json
import codecs
from functools import partial
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader


class LCQuADReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_question_len: Optional[int] = None,
                 max_logical_form_len: Optional[int] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_question_len = max_question_len
        self.max_logical_form_len = max_logical_form_len

    @overrides
    def text_to_instance(self, question_tokens: List[Token], logical_form: str) -> Instance:
        fields = dict()
        question_field = TextField(question_tokens, self.token_indexers)
        fields["question"] = question_field

        if logical_form is None:
            logical_form = np.zeros(self.max_logical_form_len)
        # TODO parse and store logical form actions sequences
        logical_form_field = logical_form
        fields["logical_form"] = logical_form_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with codecs.open(file_path) as fp:
            dataset = json.load(file_path)

        for sample in dataset:
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(sample["question"])],
                sample["logical_form"]
            )