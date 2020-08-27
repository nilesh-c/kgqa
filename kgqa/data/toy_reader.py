import codecs
import json
import dill as pickle
from typing import *

from allennlp.data import Instance, Field
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ListField, IndexField, ProductionRuleField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.semparse import ParsingError
from overrides import overrides
from pathlib import Path
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from kgqa.semparse.context.lcquad_context import LCQuADContext
from kgqa.semparse.executor.executor import Executor
from kgqa.semparse.language import LCQuADLanguage


class ToyReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy=False)

        self.tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer(namespace='target_tokens')}

    @overrides
    def text_to_instance(self, source_tokens: List[Token],
                         target_tokens: List[Token]) -> Optional[Instance]:
        source_tokens.insert(0, Token(START_SYMBOL))
        source_tokens.append(Token(END_SYMBOL))

        target_tokens.insert(0, Token(START_SYMBOL))
        target_tokens.append(Token(END_SYMBOL))

        fields = {
            'source_tokens': TextField(source_tokens, self.token_indexers),
            'target_tokens': TextField(target_tokens, self.target_token_indexers),
        }

        return Instance(fields)


    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with codecs.open(file_path) as fp:
            dataset = fp.readlines()

        for line in dataset:
            source, target = line.split("|")

            instance = self.text_to_instance(
                self.tokenizer.tokenize(source),
                self.tokenizer.tokenize(target),
            )

            yield instance
