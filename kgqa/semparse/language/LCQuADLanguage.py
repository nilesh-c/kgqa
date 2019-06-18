import types
from functools import wraps
from collections import UserDict, MutableMapping, defaultdict
from typing import Set, Union, Optional, Callable, TypeVar, Dict, List, Tuple, Any, Iterable
from numbers import Number

from hdt import IdentifierPosition

from kgqa.semparse.context import HdtQAContext

try:
    from allennlp.semparse.domain_languages.domain_language import DomainLanguage, PredicateType, predicate
except:
    from allennlp.semparse.domain_languages.domain_language import DomainLanguage, PredicateType, predicate

class Predicate(str):
    pass

class ReversedPredicate(Predicate):
    pass

class Entity(str):
    pass

class ResultSet:
    pass

class EntityResultSet(ResultSet):
    def __init__(self, entity: Entity):
        self.entity = entity

class GraphPatternResultSet(ResultSet):
    def __init__(self, patterns: Set[Tuple[Any, Predicate, Any]]):
        self.patterns = patterns

class Count:
    def __init__(self, result_set: GraphPatternResultSet):
        self.result_set = result_set

class Contains:
    def __init__(self, subset: ResultSet, superset: ResultSet):
        self.superset = superset
        self.subset = subset

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

class LCQuADLanguage(DomainLanguage):
    """
    Implements the functions in our custom variable-free functional query language for the LCQuAD dataset.

    """
    def __init__(self, context: HdtQAContext):
        self.magic_replace = [(",", "MAGIC_COMMA"),
                 ("(", "MAGIC_LEFT_PARENTHESIS"),
                 (")", "MAGIC_RIGHT_PARENTHESIS")]
        self.context = context
        start_types = {Number, Entity, Predicate}
        functions, function_types = make_function_dicts()
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
        self.call_stack: List[Tuple] = []

    def get_new_variable(self) -> Entity:
        self.var_counter += 1
        return self.var_counter

    def query_hdt(self, result_set: GraphPatternResultSet):
        fix_sub = lambda x: f"?{x}" if isinstance(x, int) else self.context.verify_uri(x, IdentifierPosition.Subject)
        fix_obj = lambda x: f"?{x}" if isinstance(x, int) else self.context.verify_uri(x, IdentifierPosition.Object)

        query = [(fix_sub(p[0]), p[1], fix_obj(p[2])) for p in result_set.patterns]

        out_var = f"?{self.var_stack.pop()}"
        return set(self.context.join(query, out_var))

    def execute(self, logical_form: str) -> Union[Iterable[str], bool, int]:
        self.reset_state()
        result: GraphPatternResultSet = super().execute(logical_form)

        out = None
        if isinstance(result, GraphPatternResultSet):
            out = self.query_hdt(result)

        elif isinstance(result, Contains):
            superset, subset = result.superset, result.subset

            if isinstance(superset, EntityResultSet) and isinstance(subset, EntityResultSet):
                superset = {str(superset.entity)}
                subset = {str(subset.entity)}

            elif isinstance(superset, GraphPatternResultSet):
                if isinstance(subset, GraphPatternResultSet):
                    subset = self.query_hdt(subset)

                elif isinstance(subset, EntityResultSet):
                    subset = {str(subset.entity)}

                superset = self.query_hdt(superset)
                if not subset:
                    if not superset:
                        print("WARNING: both empty sets in contains(superset, subset)")
                    else:
                        print("WARNING: empty subset in contains(superset, subset)")

            out = subset.issubset(superset)

        elif isinstance(result, Count):
            out = self.query_hdt(result.result_set)
            out = len(out)

        return out

    def replace_var(self, replace_func):
        def func(pattern):
            s, p, o = pattern
            if isinstance(s, int):
                s = replace_func(s)
            if isinstance(o, int):
                o = replace_func(o)

            return s, p, o

        return func

    def reverse_check(self, pattern: Tuple[Any, Predicate, Any]) -> Tuple[Any, Predicate, Any]:
        s, p, o = pattern
        if isinstance(p, ReversedPredicate):
            pattern = (o, Predicate(p), s)
        return pattern

    @record_call
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

    @record_call
    @predicate
    def intersection(self, intermediate_results1: ResultSet, intermediate_results2: ResultSet) -> ResultSet:
        """
        Return intersection of two sets of entities.
        """
        assert isinstance(intermediate_results1, GraphPatternResultSet)
        assert isinstance(intermediate_results2, GraphPatternResultSet)
        popped_var1 = self.var_stack.pop()
        popped_var2 = self.var_stack.pop()
        popped_patterns1 = self.pattern_stack.pop().patterns
        popped_patterns2 = self.pattern_stack.pop().patterns

        # replace the higher numbered var with the lower var
        lesser = min(popped_var1, popped_var2)
        higher = max(popped_var1, popped_var2)

        replace = lambda x: lesser if x == higher else x

        replace_func = self.replace_var(replace)

        popped_patterns1 = set(map(replace_func, popped_patterns1))
        popped_patterns2 = set(map(replace_func, popped_patterns2))

        gpset = GraphPatternResultSet(popped_patterns1 | popped_patterns2)

        self.var_stack.append(lesser)
        self.pattern_stack.append(gpset)

        return gpset

    @record_call
    @predicate
    def get(self, entity: Entity) -> ResultSet:
        """
        Get entity and wrap it in a set.
        """
        for original, replace in self.magic_replace:
            entity = entity.replace(replace, original)
        return EntityResultSet(entity)

    @record_call
    @predicate
    def reverse(self, predicate: Predicate) -> Predicate:
        """
        Return the reverse of given predicate.
        """
        return ReversedPredicate(predicate)

    @record_call
    @predicate
    def count(self, intermediate_results: ResultSet) -> Count:
        """
        Returns a count of a set of entities.
        """
        return Count(intermediate_results)

    @record_call
    @predicate
    def contains(self, superset: ResultSet, subset: ResultSet) -> Contains:
        """
        Returns a boolean value indicating whether subset is "contained" inside superset
        """
        return Contains(subset, superset)


if __name__ == '__main__':
    ctx = HdtQAContext('/data/nilesh/datasets/dbpedia/hdt/dbpedia2016-04en.hdt')
    l = LCQuADLanguage(ctx)
    # ers = l.execute('(find (get http://dbpedia.org/resource/Barack_Obama), (reverse http://dbpedia.org/ontology/religion))')
    # ers = l.execute('(intersection (find '
    #                 '(find (get http://dbpedia.org/resource/Barack_Obama),(reverse http://dbpedia.org/ontology/religion)),'
    #                 'http://dbpedia.org/ontology/religion), (find (get http://dbpedia.org/class/yago/Doctor110020890) http://www.w3.org/1999/02/22-rdf-syntax-ns#type))')

    # ers = l.execute("(contains (find (get http://dbpedia.org/resource/Edward_Tuckerman), (reverse http://www.w3.org/1999/02/22-rdf-syntax-ns#type)), (find (get http://dbpedia.org/resource/Barack_Obama), (reverse http://www.w3.org/1999/02/22-rdf-syntax-ns#type)))")
    # ers = l.execute('(intersection (find (get http://dbpedia.org/resource/Protestantism), http://dbpedia.org/ontology/religion), (find (get http://dbpedia.org/resource/Transylvania_University) http://dbpedia.org/ontology/education))')

    # print(l.logical_form_to_action_sequence("(find (find E2 P3) P4)"))
    # ers = l.execute("(find (get E2) (reverse P3))")
    # ers = l.execute('(find (find (get E2), P3), P4)')
    # ers = l.execute('(intersection (find (get E2), P2), (find (get E1), P1))')

    import codecs, json
    from tqdm import tqdm
    import traceback
    import time

    start_time = time.time()

    result_count = 0
    template = "/data/nilesh/datasets/LC-QuAD/lcquad.annotated.funq.{}.json"
    for split in ['train']:
        newdataset = []
        with codecs.open(template.format(split)) as fp:
            data = json.load(fp)
            for doc in tqdm(data):
                # try:
                query_time = time.time()
                results = l.execute(doc['logical_form'])
                if results:
                    if isinstance(results, bool) or isinstance(results, int):
                        results = [results]
                    else:
                        results = list(results)
                else:
                    results = []
                query_time = time.time() - query_time
                result_count += len(results)
                doc['results'] = results
                doc['time'] = query_time
                newdataset.append(doc)
                # except Exception as err:
                #     print("ERROR")
                #     traceback.print_tb(err.__traceback__)
                #     print("\n\n", doc['logical_form'])

        print("--- %s seconds ---" % (time.time() - start_time))

        with codecs.open(template.format(f"{split}.results"), "w") as fp:
            json.dump(newdataset, fp, indent=4, separators=(',', ': '), sort_keys=True)