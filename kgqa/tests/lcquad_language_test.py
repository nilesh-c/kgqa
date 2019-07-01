import time
from unittest import TestCase
from typing import List

from kgqa.common.testcase import KGQATestCase
from kgqa.semparse.context.lcquad_context import LCQuADContext
from kgqa.semparse.language import LCQuADLanguage


class LCQuADLanguageTest(KGQATestCase):
    @classmethod
    def setUpClass(cls):
        # cls.setUpStubExecutor()
        cls.setUpExecutor()
        cls.test_data = []

        logical_form = """(find (find (get http://dbpedia.org/resource/Li_Si) http://dbpedia.org/ontology/successor)
                                (reverse http://dbpedia.org/ontology/monarch))"""
        question_entities = ['http://dbpedia.org/resource/Li_Si']
        question_predicates = ['http://dbpedia.org/ontology/successor', 'http://dbpedia.org/ontology/monarch']
        expected_result = {'http://dbpedia.org/resource/King_Zhuangxiang_of_Qin',
                           'http://dbpedia.org/resource/Qin_Shi_Huang'}
        cls.test_data.append((logical_form, question_entities, question_predicates, expected_result))

        # intersection of two one-hop traversals and then one more hop
        # includes entity containing MAGIC string
        logical_form = """(find (intersection (find (get http://dbpedia.org/ontology/Automobile)
                                https://www.w3.org/1999/02/22-rdf-syntax-ns#type) (find (get
                                http://dbpedia.org/resource/BroadmeadowsMAGIC_COMMA_Victoria)
                                http://dbpedia.org/property/assembly)) (reverse http://dbpedia.org/ontology/parentCompany))"""
        question_entities = ['http://dbpedia.org/ontology/Automobile',
                             'http://dbpedia.org/resource/BroadmeadowsMAGIC_COMMA_Victoria']
        question_predicates = ['http://dbpedia.org/property/assembly',
                               'https://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                               'http://dbpedia.org/ontology/parentCompany']
        expected_result = {'http://dbpedia.org/resource/Ford_Motor_Company'}
        cls.test_data.append((logical_form, question_entities, question_predicates, expected_result))

        # intersection of intersection
        logical_form = """(intersection (intersection (find (get http://dbpedia.org/resource/Ernest_Rutherford)
                                http://dbpedia.org/ontology/doctoralAdvisor) (find (get 
                                http://dbpedia.org/resource/Charles_Drummond_Ellis) http://dbpedia.org/property/doctoralStudents))
                                (find (get http://dbpedia.org/ontology/Scientist) https://www.w3.org/1999/02/22-rdf-syntax-ns#type))"""
        question_entities = ['http://dbpedia.org/resource/Ernest_Rutherford',
                             'http://dbpedia.org/resource/Charles_Drummond_Ellis',
                             'http://dbpedia.org/ontology/Scientist']
        question_predicates = ['https://www.w3.org/1999/02/22-rdf-syntax-ns#type',
                               'http://dbpedia.org/property/doctoralStudents',
                               'http://dbpedia.org/ontology/doctoralAdvisor']
        expected_result = {'http://dbpedia.org/resource/James_Chadwick'}
        cls.test_data.append((logical_form, question_entities, question_predicates, expected_result))

        # count query involving intersection
        logical_form = """(count (intersection (find (get http://dbpedia.org/resource/HBO) http://dbpedia.org/ontology/company)
                                (find (get http://dbpedia.org/ontology/TelevisionShow) https://www.w3.org/1999/02/22-rdf-syntax-ns#type)))"""
        question_entities = ['http://dbpedia.org/resource/HBO', 'http://dbpedia.org/ontology/TelevisionShow']
        question_predicates = ['http://dbpedia.org/ontology/company',
                               'https://www.w3.org/1999/02/22-rdf-syntax-ns#type']
        expected_result = 38
        cls.test_data.append((logical_form, question_entities, question_predicates, expected_result))

        # simple boolean query
        logical_form = """(contains (find (get http://dbpedia.org/resource/Kevin_Jonas) http://dbpedia.org/ontology/formerBandMember)
                                (get http://dbpedia.org/resource/Jonas_Brothers))"""
        question_entities = ['http://dbpedia.org/resource/Jonas_Brothers', 'http://dbpedia.org/resource/Kevin_Jonas']
        question_predicates = ['http://dbpedia.org/ontology/formerBandMember']
        expected_result = True
        cls.test_data.append((logical_form, question_entities, question_predicates, expected_result))

    def setUp(self):
        super().setUp()
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f" % (self.id(), t))

    def _get_language(self, question_entities: List[str], question_predicates: List[str]) -> LCQuADLanguage:
        context = LCQuADContext(self.executor,
                                question_tokens=[],
                                question_entities=question_entities,
                                question_predicates=question_predicates)
        language = LCQuADLanguage(context)
        return language

    def _run_execute_test(self, logical_form: str, question_entities: List[str], question_predicates: List[str]):
        language = self._get_language(question_entities, question_predicates)
        return language.execute(logical_form)

    def _run_logical_form_to_action_sequence_test(self, logical_form: str, question_entities: List[str], question_predicates: List[str]):
        language = self._get_language(question_entities, question_predicates)
        language.all_possible_productions()
        action_sequences = language.logical_form_to_action_sequence(logical_form)
        return language.execute_action_sequence(action_sequences)

    def test_logical_form_to_action_sequence(self):
        for logical_form, question_entities, question_predicates, expected_result in self.test_data:
            results = self._run_logical_form_to_action_sequence_test(logical_form, question_entities, question_predicates)
            assert results == expected_result

    def test_execute(self):
        for logical_form, question_entities, question_predicates, expected_result in self.test_data:
            results = self._run_execute_test(logical_form, question_entities, question_predicates)
            assert results == expected_result
