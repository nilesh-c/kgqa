from typing import Iterable, Tuple, List, Optional

from kgqa.semparse.executor.executor import Executor


class StubExecutor(Executor):
    def triples(self, subject: Optional[str], predicate: Optional[str], object: Optional[str]) -> Iterable:
        pass

    def join(self, patterns: List[Tuple[str, str, str]], outvar: Optional[int]) -> Iterable:
        pass

    def subjects(self, predicate: str, object: str) -> Iterable[str]:
        pass

    def predicates(self, subject: str, object: str) -> Iterable[str]:
        pass

    def objects(self, subject: str, predicate: str) -> Iterable[str]:
        pass

    def subject_predicates(self, object: str) -> Iterable[Tuple[str, str]]:
        pass

    def subject_objects(self, predicate: str) -> Iterable[Tuple[str, str]]:
        pass

    def predicate_objects(self, subject: str) -> Iterable[Tuple[str, str]]:
        pass