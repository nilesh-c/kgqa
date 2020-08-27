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
from kgqa.semparse.executor import StubExecutor
from kgqa.semparse.executor.executor import Executor
from kgqa.semparse.language import LCQuADLanguage

magic_replace = [(",", "MAGIC_COMMA"),
                 ("(", "MAGIC_LEFT_PARENTHESIS"),
                 (")", "MAGIC_RIGHT_PARENTHESIS")]

def deurify_predicate(uri: str):
    #return uri
    last = uri.split("/")[-1]
    # return '-' + last if uri.startswith('-') else last
    return last[1:] if last.startswith('-') else last


class LCQuADReaderSimple(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 predicates: List[str] = None,
                 ontology_types: List[str] = None):
        super().__init__(lazy=False)

        self.tokenizer = tokenizer or WordTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer(namespace='target_tokens')}

        self.predicates = [deurify_predicate(p) for p in predicates]
        self.original_predicates = predicates
        self.unique_predicates = list(set(self.predicates))
        self.ontology_types = ontology_types
        self.executor = StubExecutor()
        context = LCQuADContext(self.executor, [], ['ENT_1', 'ENT_2'], self.unique_predicates)
        self.language = LCQuADLanguage(context)

    @overrides
    def text_to_instance(self, question_entities: List[str],
                         question_tokens: List[Token],
                         logical_form: str) -> Optional[Instance]:
        try:
            if any([True for i in self.ontology_types if i in logical_form]):
                return None

            # if "intersection" in logical_form or "contains" in logical_form or "count" in logical_form:
            #     return None

            for old, new in zip(self.original_predicates, self.predicates):
                logical_form = logical_form.replace('-' + old, new)
                logical_form = logical_form.replace(old, new)

            if "/" in logical_form:
                return None
            # for lang_tokens in ['findSet', 'find', 'intersection']:
            #     logical_form = logical_form.replace(lang_tokens, ' ')

            # logical_form_tokens = self.tokenizer.tokenize(logical_form)
            logical_form_tokens = [Token(i) for i in self.language.logical_form_to_action_sequence(logical_form)]

            question_tokens.insert(0, Token(START_SYMBOL))
            question_tokens.append(Token(END_SYMBOL))

            logical_form_tokens.insert(0, Token(START_SYMBOL))
            logical_form_tokens.append(Token(END_SYMBOL))

            fields = {
                'source_tokens': TextField(question_tokens, self.token_indexers),
                'target_tokens': TextField(logical_form_tokens, self.target_token_indexers),
            }

            return Instance(fields)
        except ParsingError:
            print(logical_form)
            return None

    # def _write_custom_cache(self, file_path: str, instances: List[Instance]):
    #     cache_file = file_path + ".cache"
    #     if not Path(cache_file).exists():
    #         with open(cache_file, "wb") as fp:
    #             pickle.dump(instances, fp)
    #
    # def _read_custom_cache(self, file_path: str) -> List[Instance]:
    #     cache_file = file_path + ".cache"
    #     with open(cache_file, 'rb') as fp:
    #         cached = pickle.load(fp)
    #     return cached

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        whitelist = set(['http://dbpedia.org/ontology/award',
                     'http://dbpedia.org/ontology/religion',
                     'http://dbpedia.org/property/awards',
                     'http://dbpedia.org/ontology/birthPlace',
                     'http://dbpedia.org/ontology/sport',
                     'http://dbpedia.org/ontology/team',
                     'http://dbpedia.org/ontology/manufacturer',
                     'http://dbpedia.org/ontology/deathPlace',
                     'http://dbpedia.org/ontology/almaMater'])

        def whitelisted(preds):
            return set(preds).issubset(whitelist)

        bad_count = 0
        good_count = 0

        with codecs.open(file_path) as fp:
            dataset = json.load(fp)

        for doc in dataset:
            if doc['entities']: #  and whitelisted(doc['predicates']):
                logical_form = doc["logical_form"]
                for ent in doc['entities']:
                    logical_form = logical_form.replace(ent['uri'], ent['placeholder'])
                instance = self.text_to_instance(
                    [ent['placeholder'] for ent in doc['entities']],
                    self.tokenizer.tokenize(doc["question_mapped"]),
                    logical_form
                )
                if instance:
                    good_count += 1
                    yield instance
                else:
                    bad_count += 1
            else:
                bad_count += 1

        print("BAD count:", bad_count)
        print("GOOD count:", good_count)
