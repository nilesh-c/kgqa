from allennlp.common.testing import ModelTestCase
from pathlib import Path

from kgqa.common.testcase import KGQATestCase

PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / "..").resolve()  # pylint: disable=no-member
MODULE_ROOT = PROJECT_ROOT / "dialogue_models"
TESTS_ROOT = MODULE_ROOT / "tests"


class CSQAMmlSemanticParsingTest(ModelTestCase, KGQATestCase):
    def setUp(self):
        super(CSQAMmlSemanticParsingTest, self).setUp()
        model_params_file_path = self.TEST_DATA_ROOT / "experiment.json"
        dataset_sample_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.sample.json"

        self.set_up_model(model_params_file_path, dataset_sample_file_path)

    def test_model_forward(self):
        for inst in self.dataset.instances:
            self.model(inst)

    # def test_model_can_train_save_and_load(self):
    #     self.ensure_model_can_train_save_and_load(self.param_file)
