from unittest import TestCase
import pathlib

from hdt import HDTDocument

from kgqa.semparse.executor.stub_executor import StubExecutor
from kgqa.semparse.executor.hdt_executor import HdtExecutor

class KGQATestCase(TestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()  # pylint: disable=no-member
    MODULE_ROOT = PROJECT_ROOT / "kgqa"
    TEST_DATA_ROOT = MODULE_ROOT / "tests" / "data"

    @classmethod
    def setUpExecutor(self):
        hdt = HDTDocument('/data/nilesh/datasets/dbpedia/hdt/dbpedia2016-04en.hdt', map=True, progress=True)
        self.executor = HdtExecutor(graph=hdt)

    @classmethod
    def setUpStubExecutor(self):
        self.executor = StubExecutor()