import json
from functools import wraps
from typing import Optional, Iterable, Tuple, List
import redis
import hdt
from hdt import HDTDocument, IdentifierPosition


def cached(func):
    """
    Decorator that caches the results of the function call.

    We use Redis in this example, but any cache (e.g. memcached) will work.
    We also assume that the result of the function can be seralized as JSON,
    which obviously will be untrue in many situations. Tweak as needed.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.redis:
            # Generate the cache key from the function's arguments.
            key_parts = func.__name__ + json.dumps(list(args))
            key = '-'.join(key_parts)
            result = self.redis.get(key)

            if result is None:
                # Run the function and cache the result for next time.
                value = func(self, *args, **kwargs)
                value_json = json.dumps(value)
                self.redis.set(key, value_json)
            else:
                # Skip the function entirely and use the cached value instead.
                value_json = result.decode('utf-8')
                value = json.loads(value_json)
        else:
            value = func(self, *args, **kwargs)

        return value

    return wrapper


class HdtQAContext:
    def __init__(self, hdt_path: Optional[str] = None, graph: Optional[HDTDocument] = None, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        if graph:
            self.graph = graph
        else:
            self.graph = HDTDocument(hdt_path, map=False, progress=True)

    def triples(self, subject: Optional[str]='', predicate: Optional[str]='', object: Optional[str]='')\
            -> Tuple[hdt.JoinIterator, int]:
        """
        Generator over the triple store
        Returns triples that match the given triple pattern and the count.
        """
        result_iter, count = self.graph.search_triples(subject, predicate, object)
        return list(result_iter), count

    @cached
    def join(self, patterns: List[Tuple[str, str, str]], outvar: Optional[str] = None) -> Iterable:
        """
        Joins a list of basic graph patterns and
        returns triples that match multiple triple patterns.
        """
        result_iter = self.graph.search_join(patterns)
        if outvar:
            return [uri for join_set in result_iter for var, uri in join_set if var == outvar]
        else:
            return list(result_iter)

    def subjects(self, predicate=None, object=None) -> Iterable[str]:
        """
        A generator of subjects with the given predicate and object
        """
        return [s for s, p, o in self.triples(predicate=predicate, object=object)[0]]

    def predicates(self, subject=None, object=None) -> Iterable[str]:
        """
        A generator of predicates with the given subject and object
        """
        return [p for s, p, o in self.triples(subject=subject, object=object)[0]]

    def objects(self, subject=None, predicate=None) -> Iterable[str]:
        """
        A generator of objects with the given subject and predicate
        """
        return [o for s, p, o in self.triples(subject=subject, predicate=predicate)[0]]

    def subject_predicates(self, object=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (subject, predicate) tuples for the given object
        """
        return [(s, p) for s, p, o in self.triples(object=object)[0]]

    def subject_objects(self, predicate=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (subject, object) tuples for the given predicate
        """
        return [(s, o) for s, p, o in self.triples(predicate=predicate)[0]]

    def predicate_objects(self, subject=None) -> Iterable[Tuple[str, str]]:
        """
        A generator of (predicate, object) tuples for the given subject
        """
        return [(p, o) for s, p, o in self.triples(subject=subject)[0]]

    def verify_uri(self, uri: str, position: IdentifierPosition) -> Optional[str]:
        uri = uri.replace("'", "")
        sub_id = self.graph.convert_term(uri, position)
        if not sub_id:
            uri = ascii(uri.encode())[2:-1].replace("\\x", "x")
            sub_id = self.graph.convert_term(uri, position)

        return uri if sub_id else None
