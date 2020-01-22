import codecs
import json
import dill as pickle
from typing import *

from allennlp.data import Instance, Field
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ListField, IndexField, ProductionRuleField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.semparse import ParsingError
from overrides import overrides
from pathlib import Path


from kgqa.semparse.context.lcquad_context import LCQuADContext
from kgqa.semparse.executor.executor import Executor
from kgqa.semparse.language import LCQuADLanguage

magic_replace = [(",", "MAGIC_COMMA"),
                 ("(", "MAGIC_LEFT_PARENTHESIS"),
                 (")", "MAGIC_RIGHT_PARENTHESIS")]




class LCQuADReader(DatasetReader):
    def __init__(self, executor: Executor,
                 tokenizer: Callable[[str], List[str]] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 predicates: List[str] = None) -> None:
        super().__init__(lazy=False)

        def splitter(x: str):
            return [w.text for w in
                    SpacyWordSplitter(language='en_core_web_sm',
                                      pos_tags=False).split_words(x)]
        self.tokenizer = tokenizer or splitter
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.predicates = predicates
        self.executor = executor

    @overrides
    def text_to_instance(self, question_tokens: List[Token],
                         logical_form: str,
                         question_entities: List[str],
                         question_predicates: List[str]) -> Optional[Instance]:
        try:
            context = LCQuADContext(self.executor, question_tokens, question_entities, question_predicates)
            language = LCQuADLanguage(context)
            # print("CONSTANT:" + str(language._functions['http://dbpedia.org/ontology/creator']))
            target_action_sequence = language.logical_form_to_action_sequence(logical_form)
            labelled_results = language.execute_action_sequence(target_action_sequence)
            if isinstance(labelled_results, set) and len(labelled_results) > 1000:
                return None

            production_rule_fields = [ProductionRuleField(rule, is_global_rule=True) for rule
                                      in language.all_possible_productions()]
            action_field = ListField(production_rule_fields)
            action_map = {action.rule: i for i, action in enumerate(production_rule_fields)}
            target_action_sequence_field = ListField([ListField([IndexField(action_map[a], action_field)
                                                      for a in target_action_sequence])])

            fields = {
                'question': TextField(question_tokens, self.token_indexers),
                'question_entities': MetadataField(question_entities),
                'question_predicates': MetadataField(question_predicates),
                'world': MetadataField(language),
                'actions': action_field,
                'target_action_sequences': target_action_sequence_field,
                'logical_forms': MetadataField([logical_form]),
                'labelled_results': MetadataField(labelled_results),
            }

            return Instance(fields)
        except ParsingError:
            return None

    def _write_custom_cache(self, file_path: str, instances: List[Instance]):
        def remove_executor(instance: Instance):
            language: LCQuADLanguage = instance.fields['world'].metadata
            language.context.executor = None
            language.executor = None
            return instance

        cache_file = file_path + ".cache"
        if not Path(cache_file).exists():
            instances = [remove_executor(i) for i in instances]
            with open(cache_file, "wb") as fp:
                pickle.dump(instances, fp)

    def _read_custom_cache(self, file_path: str) -> List[Instance]:
        def add_executor(instance: Instance):
            language: LCQuADLanguage = instance.fields['world'].metadata
            language.context.executor = self.executor
            language.executor = self.executor
            return instance

        cache_file = file_path + ".cache"
        with open(cache_file, 'rb') as fp:
            cached = pickle.load(fp)
        for i in cached:
            yield add_executor(i)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        try:
            for i in self._read_custom_cache(file_path):
                yield i
        except FileNotFoundError as e:
            instances = []
            bad_count = 0
            good_count = 0

            with codecs.open(file_path) as fp:
                dataset = json.load(fp)

            for doc in dataset:
                if doc['entities']:
                    instance = self.text_to_instance(
                        [Token(x) for x in self.tokenizer(doc["question"])],
                        doc["logical_form"],
                        [entity['uri'] for entity in doc['entities']],
                        doc.get('predicate_candidates', self.predicates)
                    )
                    if instance:
                        good_count += 1
                        instances.append(instance)
                        yield instance
                    else:
                        bad_count += 1
                else:
                    bad_count += 1

            self._write_custom_cache(file_path, instances)

            print("BAD count:", bad_count)
            print("GOOD count:", good_count)
