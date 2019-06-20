import json
from collections import defaultdict
from functools import wraps
from typing import TypeVar, MutableMapping, Optional, Callable

from allennlp.semparse.domain_languages.domain_language import PredicateType

from kgqa.semparse.language.lcquad_language import Entity, Predicate, EntityResultSet, GraphPatternResultSet, ResultSet

_KT = TypeVar('_KT')
_VT = TypeVar('_VT')


class FuncDict(MutableMapping):
    def __init__(self, store: MutableMapping,
                 get_func: Optional[Callable[[_KT], _VT]] = None,
                 contains_func: Optional[Callable[[_KT], _VT]] = None,
                 *args, **kwargs) -> None:
        self.store = store
        self.update(dict(*args, **kwargs))
        self.get_func = get_func
        self.contains_func = contains_func

    def __getitem__(self, key: _KT) -> _VT:
        result = None
        if self.get_func:
            result = self.get_func(key)

        if result:
            return result
        else:
            return self.store[key]

    def __contains__(self, key: object) -> bool:
        result = None
        if self.get_func:
            result = self.contains_func(key)

        if result:
            return result
        else:
            return self.store.__contains__(key)

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


def make_function_dicts():
    entity_type = PredicateType.get_type(Entity)
    predicate_type = PredicateType.get_type(Predicate)

    def is_entity(uri: str):
        return uri.startswith("http://dbpedia.org/")

    def is_predicate(uri: str):
        return uri.startswith("htt")

    def get_value_func(key):
        if is_entity(key):
            return lambda: Entity(key)
        elif is_predicate(key):
            return lambda: Predicate(key)
        else:
            return None

    def get_type_func(key):
        if is_entity(key):
            return [entity_type]
        elif is_predicate(key):
            return [predicate_type]
        else:
            return None

    def contains_func(key):
        if is_entity(key):
            # return 0 < int(key[1:]) <= max_entity_id
            return True
        elif is_predicate(key):
            # return 0 < int(key[1:]) <= max_predicate_id
            return True
        else:
            return None

    return FuncDict(dict(), get_value_func, contains_func), FuncDict(defaultdict(list), get_type_func, contains_func)


def record_call(func):
    def parse_arg(arg):
        if isinstance(arg, EntityResultSet):
            return arg.entity
        elif isinstance(arg, GraphPatternResultSet):
            return arg.patterns
        else:
            return arg

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        results: ResultSet = func(self, *args, **kwargs)
        args = [parse_arg(arg) for arg in args]
        pushed_var = self.var_stack[-1] if self.var_stack else None
        self.call_stack.append((func.__name__, pushed_var, args, parse_arg(results)))
        return results
    return wrapper

def cached(func):
    """
    Decorator that caches the results of a method call.
    Set self.cache to a redis.Redis instance.

    Assumptions:
    1. Arguments and result of the function can be seralized as JSON,
    2. self.cache has .get() and .set() methods
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.cache:
            # Generate the cache key from the function's arguments.
            key_parts = func.__name__ + json.dumps(list(args))
            key = '-'.join(key_parts)
            result = self.cache.get(key)

            if result is None:
                # Run the function and cache the result for next time.
                value = func(self, *args, **kwargs)
                value_json = json.dumps(value)
                self.cache.set(key, value_json)
            else:
                # Skip the function entirely and use the cached value instead.
                value_json = result.decode('utf-8')
                value = json.loads(value_json)
        else:
            value = func(self, *args, **kwargs)

        return value

    return wrapper
