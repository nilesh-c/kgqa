from unittest import TestCase
import pathlib

from hdt import HDTDocument
from kgqa.semparse.executor.hdt_executor import HdtExecutor

class KGQATestCase(TestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()  # pylint: disable=no-member
    MODULE_ROOT = PROJECT_ROOT / "dialogue_models"
    TOOLS_ROOT = MODULE_ROOT / "tools"
    TESTS_ROOT = MODULE_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"

    @classmethod
    def setUpClass(self):
        hdt = HDTDocument('/data/nilesh/datasets/dbpedia/hdt/dbpedia2016-04en.hdt', map=True, progress=True)
        self.executor = HdtExecutor(graph=hdt)