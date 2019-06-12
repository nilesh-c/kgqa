import hdt
from typing import Optional, Iterable, Tuple, List
from hdt import HDTDocument


class HdtQAContext:
    def __init__(self, hdt_path: str):
        self.graph = HDTDocument(hdt_path)

    def triples(self, subject: Optional[str]='', predicate: Optional[str]='', object: Optional[str]='')\
            -> Tuple[hdt.JoinIterator, int]:
        """
        Generator over the triple store
        Returns triples that match the given triple pattern and the count.
        """
        return self.graph.search_triples(subject, predicate, object)

    def join(self, patterns: List[Tuple[str, str, str]]) -> hdt.JoinIterator:
        """
        Joins a list of basic graph patterns and
        returns triples that match multiple triple patterns.
        """
        return self.graph.search_join(patterns)

    def subjects(self, predicate=None, object=None) -> Iterable[str]:
        """
        A generator of subjects with the given predicate and object
        """
        for s, p, o in self.triples(predicate=predicate, object=object)[0]:
            yield s

    def predicates(self, subject=None, object=None) -> Iterable[str]:
        """
        A generator of predicates with the given subject and object
        """
        for s, p, o in self.triples(subject=subject, object=object)[0]:
            yield p

    def objects(self, subject=None, predicate=None) -> Iterable[str]:
        """
        A generator of objects with the given subject and predicate
        """
        for s, p, o in self.triples(subject=subject, predicate=predicate)[0]:
            yield o

    def subject_predicates(self, object=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (subject, predicate) tuples for the given object
        """
        for s, p, o in self.triples(object=object)[0]:
            yield s, p

    def subject_objects(self, predicate=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (subject, object) tuples for the given predicate
        """
        for s, p, o in self.triples(predicate=predicate)[0]:
            yield s, o

    def predicate_objects(self, subject=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (predicate, object) tuples for the given subject
        """
        for s, p, o in self.triples(subject=subject)[0]:
            yield p, o

