import codecs
import json
from typing import *

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField, ListField, IndexField, ProductionRuleField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from kgqa.semparse.context.lcquad_context import LCQuADContext
from kgqa.semparse.executor.executor import Executor
from kgqa.semparse.language import LCQuADLanguage
from kgqa.semparse.language.lcquad_language import Predicate


class LCQuADReader(DatasetReader):
    def __init__(self, executor: Executor,
                 tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_question_len: Optional[int] = None,
                 max_logical_form_len: Optional[int] = None,
                 predicates: List[str] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_question_len = max_question_len
        self.max_logical_form_len = max_logical_form_len
        self.predicates = predicates
        self.executor = executor

    @overrides
    def text_to_instance(self, question_tokens: List[Token],
                         logical_form: str,
                         question_entities: List[str],
                         question_predicates: List[str]) -> Instance:
        context = LCQuADContext(self.executor, question_tokens, question_entities, question_predicates)
        language = LCQuADLanguage(context)
        print("CONSTANT:" + str(language._functions['http://dbpedia.org/ontology/creator']))
        target_action_sequence = language.logical_form_to_action_sequence(logical_form)

        production_rule_fields = [ProductionRuleField(rule, is_global_rule=True) for rule
                                  in language.all_possible_productions()]
        action_field = ListField(production_rule_fields)
        action_map = {action.rule: i for i, action in enumerate(production_rule_fields)}
        target_action_sequence_field = ListField([IndexField(action_map[a], action_field)
                                                  for a in target_action_sequence])

        fields = {
            'question': TextField(question_tokens, self.token_indexers),
            'question_entities': MetadataField(question_entities),
            'question_predicates': MetadataField(question_predicates),
            'world': MetadataField(language),
            'actions': action_field,
            'target_action_sequence': target_action_sequence_field
        }

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        with codecs.open(file_path) as fp:
            dataset = json.load(file_path)

        for doc in dataset:
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(doc["question"])],
                doc["logical_form"],
                [entity['uri'] for entity in doc['entities']],
                doc.get('predicate_candidates', self.predicates)
            )
