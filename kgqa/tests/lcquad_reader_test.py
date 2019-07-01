import codecs
import json

from allennlp.data import Token

from kgqa.common.testcase import KGQATestCase
from kgqa.data.lcquad_reader import LCQuADReader

class LCQuADReaderTest(KGQATestCase):


    @classmethod
    def setUpClass(self):
        super().setUpClass()
        self.setUpStubExecutor()
        # predicates_file_path = self.TEST_DATA_ROOT / "predicates.txt"
        predicates_file_path = self.TEST_DATA_ROOT / "properties.txt"
        dataset_sample_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.sample.json"

        with codecs.open(predicates_file_path) as fp:
            self.predicates = [i.strip() for i in fp]

        with codecs.open(dataset_sample_file_path) as fp:
            self.dataset = json.load(fp)[:3]

    def test_text_to_instance(self):
        dbo_classes = set([dbo for dbo in self.predicates if dbo.split("/")[-1][0].isupper()])
        binary_predicates = set(self.predicates) - dbo_classes

        reader = LCQuADReader(executor=self.executor, predicates=self.predicates)
        doc = self.dataset[0]
        # print(doc["logical_form"])
        # print([entity['uri'] for entity in doc['entities']])
        # print(doc.get('predicate_candidates', self.predicates))
        blah = reader.text_to_instance(
            [Token(x) for x in reader.tokenizer(doc["question"])],
            doc["logical_form"],
            [entity['uri'] for entity in doc['entities']],
            doc.get('predicate_candidates', self.predicates)
        )
        print(blah)