from typing import Set, Union, List, Tuple, Any, Iterable

from hdt import IdentifierPosition, HDTDocument

from kgqa.semparse.executor.hdt_executor import HdtExecutor
from kgqa.semparse.context.lcquad_context import LCQuADContext
from kgqa.semparse.executor.executor import Executor
from kgqa.semparse.util import record_call

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


class LCQuADLanguage(DomainLanguage):
    """
    Implements the functions in our custom variable-free functional query language for the LCQuAD dataset.

    """
    def __init__(self, context: LCQuADContext):
        self.magic_replace = [(",", "MAGIC_COMMA"),
                 ("(", "MAGIC_LEFT_PARENTHESIS"),
                 (")", "MAGIC_RIGHT_PARENTHESIS")]
        self.context = context
        self.executor: Executor = context.executor
        super().__init__(start_types={Entity})

        for predicate in context.question_predicates:
            self.add_constant(predicate, Predicate(predicate), type_=Predicate)

        for entity in context.question_entities:
            self.add_constant(entity, Entity(entity), type_=Entity)

        self.var_counter = 0
        self.var_stack: List[int] = []
        self.pattern_stack: List[GraphPatternResultSet] = []

    def reset_state(self):
        self.var_counter = 0
        self.var_stack: List[int] = []
        self.pattern_stack: List[GraphPatternResultSet] = []

    def get_new_variable(self) -> int:
        self.var_counter += 1
        return self.var_counter

    def query_hdt(self, result_set: GraphPatternResultSet):
        out_var = self.var_stack.pop()
        fix_sub = lambda x: f"?{x}" if isinstance(x, int) else x
        fix_obj = lambda x: f"?{x}" if isinstance(x, int) else x

        query = [(fix_sub(p[0]), p[1], fix_obj(p[2])) for p in result_set.patterns]

        return set(self.executor.join(list(query), out_var))

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
    hdt = HDTDocument('/data/nilesh/datasets/dbpedia/hdt/dbpedia2016-04en.hdt', map=True, progress=True)
    ctx = HdtExecutor(graph=hdt)
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

    # import codecs, json
    # from tqdm import tqdm
    # import traceback
    # import time
    #
    # start_time = time.time()
    #
    # result_count = 0
    # template = "/data/nilesh/datasets/LC-QuAD/lcquad.annotated.funq.{}.json"
    # for split in ['train']:
    #     newdataset = []
    #     with codecs.open(template.format(split)) as fp:
    #         data = json.load(fp)
    #         for doc in tqdm(data):
    #             # try:
    #             query_time = time.time()
    #             results = l.execute(doc['logical_form'])
    #             if results:
    #                 if isinstance(results, bool) or isinstance(results, int):
    #                     results = [results]
    #                 else:
    #                     results = list(results)
    #             else:
    #                 results = []
    #             query_time = time.time() - query_time
    #             result_count += len(results)
    #             doc['results'] = results
    #             doc['time'] = query_time
    #             newdataset.append(doc)
    #             # except Exception as err:
    #             #     print("ERROR")
    #             #     traceback.print_tb(err.__traceback__)
    #             #     print("\n\n", doc['logical_form'])
    #
    #     print("--- %s seconds ---" % (time.time() - start_time))
    #
    #     with codecs.open(template.format(f"{split}.results"), "w") as fp:
    #         json.dump(newdataset, fp, indent=4, separators=(',', ': '), sort_keys=True)