from typing import Optional, Iterable, Tuple, List
import redis
import hdt
from hdt import HDTDocument, IdentifierPosition

from kgqa.semparse.executor.executor import Executor
from kgqa.semparse.util import cached


class HdtExecutor(Executor):
    def __init__(self, hdt_path: Optional[str] = None,
                 graph: Optional[HDTDocument] = None,
                 redis_client: Optional[redis.Redis] = None):
        self.cache = redis_client
        if graph:
            self.graph = graph
        else:
            self.graph = HDTDocument(hdt_path, map=False, progress=True)

    @cached
    def triples(self, subject: Optional[str]='',
                predicate: Optional[str]='',
                object: Optional[str]='')\
            -> Iterable:
        """
        Generator over the triple store
        Returns triples that match the given triple pattern and the count.
        """
        result_iter, count = self.graph.search_triples(subject, predicate, object)
        return list(result_iter), count

    @cached
    def join(self, patterns: List[Tuple[str, str, str]],
             outvar: Optional[str] = None) -> Iterable:
        """
        Joins a list of basic graph patterns and
        returns triples that match multiple triple patterns.
        """
        patterns = self._verify_uris(patterns)
        result_iter = self.graph.search_join(patterns)
        if outvar:
            return [uri for join_set in result_iter for var, uri in join_set if var == outvar]
        else:
            return list(result_iter)

    @cached
    def subjects(self, predicate=None, object=None) -> Iterable[str]:
        """
        A generator of subjects with the given predicate and object
        """
        return [s for s, p, o in self.triples(predicate=predicate, object=object)[0]]

    @cached
    def predicates(self, subject=None, object=None) -> Iterable[str]:
        """
        A generator of predicates with the given subject and object
        """
        return [p for s, p, o in self.triples(subject=subject, object=object)[0]]

    @cached
    def objects(self, subject=None, predicate=None) -> Iterable[str]:
        """
        A generator of objects with the given subject and predicate
        """
        return [o for s, p, o in self.triples(subject=subject, predicate=predicate)[0]]

    @cached
    def subject_predicates(self, object=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (subject, predicate) tuples for the given object
        """
        return [(s, p) for s, p, o in self.triples(object=object)[0]]

    @cached
    def subject_objects(self, predicate=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (subject, object) tuples for the given predicate
        """
        return [(s, o) for s, p, o in self.triples(predicate=predicate)[0]]

    @cached
    def predicate_objects(self, subject=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (predicate, object) tuples for the given subject
        """
        return [(p, o) for s, p, o in self.triples(subject=subject)[0]]

    def _verify_uris(self, pattern: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        return [(self._verify_uri(s, IdentifierPosition.Subject), p,
                 self._verify_uri(o, IdentifierPosition.Object)) for s, p, o in pattern]

    def _verify_uri(self, uri: str, position: IdentifierPosition) -> Optional[str]:
        if uri[0] == '?':
            return uri

        uri = uri.replace("'", "")
        sub_id = self.graph.convert_term(uri, position)
        if not sub_id:
            uri = ascii(uri.encode())[2:-1].replace("\\x", "x")
            sub_id = self.graph.convert_term(uri, position)

        return uri if sub_id else None
