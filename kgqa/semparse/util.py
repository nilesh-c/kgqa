import json
from functools import wraps
from typing import TypeVar, MutableMapping, Optional, Callable


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
