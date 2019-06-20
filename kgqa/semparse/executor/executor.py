from abc import ABC, abstractmethod
from typing import Optional, Iterable, List, Tuple

class Executor(ABC):
    @abstractmethod
    def triples(self, subject: Optional[str],
                predicate: Optional[str],
                object: Optional[str])\
            -> Iterable:
        """
        Generator over the triple store
        Returns triples that match the given triple pattern and the count.
        """
        pass

    @abstractmethod
    def join(self, patterns: List[Tuple[str, str, str]],
             outvar: Optional[int]) -> Iterable:
        """
        Joins a list of basic graph patterns and
        returns triples that match multiple triple patterns.
        """
        pass

    @abstractmethod
    def subjects(self, predicate: str, object: str) -> Iterable[str]:
        """
        A generator of subjects with the given predicate and object
        """
        pass

    @abstractmethod
    def predicates(self, subject: str, object: str) -> Iterable[str]:
        """
        A generator of predicates with the given subject and object
        """
        pass

    @abstractmethod
    def objects(self, subject: str, predicate: str) -> Iterable[str]:
        """
        A generator of objects with the given subject and predicate
        """
        pass

    @abstractmethod
    def subject_predicates(self, object: str) -> Iterable[Tuple[str, str]]:
        """
        A generator of (subject, predicate) tuples for the given object
        """
        pass

    @abstractmethod
    def subject_objects(self, predicate: str) -> Iterable[Tuple[str, str]]:
        """
        A generator of (subject, object) tuples for the given predicate
        """
        pass

    @abstractmethod
    def predicate_objects(self, subject: str) -> Iterable[Tuple[str, str]]:
        """
        A generator of (predicate, object) tuples for the given subject
        """
        pass
