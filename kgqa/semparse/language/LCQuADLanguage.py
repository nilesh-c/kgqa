import types
from collections import UserDict, MutableMapping, defaultdict

from allennlp.semparse import DomainLanguage
from allennlp.semparse import predicate
from typing import Set, Union, Optional, Callable, TypeVar, Dict, List, Tuple, Any
from numbers import Number

from allennlp.semparse.domain_languages.domain_language import PredicateType
from hdt import HDTDocument

class Predicate(str):
    pass

class ReversedPredicate(Predicate):
    pass

class Entity(str):
    pass

class ResultSet():
    pass

class EntityResultSet(ResultSet):
    def __init__(self, entity: Entity):
        self.entity = entity

class GraphPatternResultSet(ResultSet):
    def __init__(self, patterns: Set[Tuple[Any, Predicate, Any]]):
        self.patterns = patterns

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

def make_function_dicts(max_entity_id, max_predicate_id):
    entity_type = PredicateType.get_type(Entity)
    predicate_type = PredicateType.get_type(Predicate)

    def get_value_func(key):
        if key[0] == 'E':
            return lambda: Entity(key)
        elif key[0] == 'P':
            return lambda: Predicate(key)
        else:
            return None

    def get_type_func(key):
        if key[0] == 'E':
            return [entity_type]
        elif key[0] == 'P':
            return [predicate_type]
        else:
            return None

    def contains_func(key):
        if key[0] == 'E':
            return 0 < int(key[1:]) <= max_entity_id
        elif key[0] == 'P':
            return 0 < int(key[1:]) <= max_predicate_id
        else:
            return None

    return FuncDict(dict(), get_value_func, contains_func), FuncDict(defaultdict(list), get_type_func, contains_func)

class LCQuADLanguage(DomainLanguage):
    """
    Implements the functions in our custom variable-free functional query language for the LCQuAD dataset.

    """

    def __init__(self, max_entity_id, max_predicate_id):
        start_types = {Number, Entity, Predicate}
        functions, function_types = make_function_dicts(max_entity_id, max_predicate_id)
        self._functions: MutableMapping[str, Callable] = functions
        self._function_types: Dict[str, List[PredicateType]] = function_types
        self._start_types: Set[PredicateType] = set([PredicateType.get_type(type_) for type_ in start_types])
        for name in dir(self):
            if isinstance(getattr(self, name), types.MethodType):
                function = getattr(self, name)
                if getattr(function, '_is_predicate', False):
                    side_arguments = getattr(function, '_side_arguments', None)
                    self.add_predicate(name, function, side_arguments)

        # Caching this to avoid recomputing it every time `get_nonterminal_productions` is called.
        self._nonterminal_productions: Dict[str, List[str]] = None

    def reset_state(self):
        self.var_counter = 0
        self.var_stack: List[int] = []
        self.pattern_stack: List[GraphPatternResultSet] = []

    def get_new_variable(self) -> Entity:
        self.var_counter += 1
        return self.var_counter

    def execute(self, logical_form: str):
        self.reset_state()
        return super().execute(logical_form)

    def execute_resultset(self, results: ResultSet) -> EntityResultSet:
        pass

    def reverse_check(self, pattern: Tuple[Any, Predicate, Any]) -> Tuple[Any, Predicate, Any]:
        s, p, o = pattern
        if isinstance(p, ReversedPredicate):
            pattern = (o, Predicate(p), s)
        return pattern

    @predicate
    def find(self, intermediate_results: ResultSet, predicate: Predicate) -> ResultSet:
        """
        Takes an entity e and a predicate p and returns the set of entities that
        have an outgoing edge p to e.

        """
        if isinstance(intermediate_results, EntityResultSet):
            object_entity = intermediate_results.entity

            new_var = self.get_new_variable()
            new_pattern = (new_var, predicate, object_entity)
            gpset = GraphPatternResultSet({self.reverse_check(new_pattern)})

            self.var_stack.append(new_var)
            self.pattern_stack.append(gpset)
        else:
            assert isinstance(intermediate_results, GraphPatternResultSet)
            popped_var = self.var_stack.pop()
            popped_patterns = self.pattern_stack.pop().patterns

            new_var = self.get_new_variable()
            new_pattern = (new_var, predicate, popped_var)
            gpset = GraphPatternResultSet(popped_patterns | {self.reverse_check(new_pattern)})

            self.var_stack.append(new_var)
            self.pattern_stack.append(gpset)

        return gpset


    @predicate
    def intersection(self, intermediate_results1: ResultSet, intermediate_results2: ResultSet) -> ResultSet:
        """
        Return intersection of two sets of entities.
        """
        pass

    @predicate
    def get(self, entity: Entity) -> ResultSet:
        """
        Get entity and wrap it in a set.
        """
        return EntityResultSet(entity)

    @predicate
    def reverse(self, predicate: Predicate) -> Predicate:
        """
        Return the reverse of given predicate.
        """
        return ReversedPredicate(predicate)

    @predicate
    def count(self, intermediate_results: ResultSet) -> Number:
        """
        Returns a count of a set of entities.
        """
        return self.execute(intermediate_results)[1]


# if __name__ == '__main__':
#     l = LCQuADLanguage(10, 10)
#     # print(l.logical_form_to_action_sequence("(find (find E2 P3) P4)"))
#     # ers = l.execute("(find (get E2) (reverse P3))")
#     ers = l.execute('(find (find (get E2), P3), P4)')
#     # ers = l.execute('(intersection (find (get E2), P2), (find (get E1), P1))')