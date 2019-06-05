import os
import copy
import json
import codecs
import rdflib
import pickle

from functools import reduce
from typing import *
from tqdm import tqdm
from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.term import *


class LCQuADSparqlToFunQuery:
    def __init__(self, entity_to_id: Dict[str, int], predicate_to_id: Dict[str, int], lisp_style=True):
        """
        Converts a small subset of SPARQL to our custom functional query language.
        There's some ugly and hardcoded stuff in the parsing logic.
        Use only with LC-QuAD dataset (https://github.com/AskNowQA/LC-QuAD).

        :param entity_to_id: Entity uri-to-id dictionary
        :param predicate_to_id: Predicate uri-to-id dictionary
        :param lisp_style: Toggle between LISP-style vs FunQL-style language
        """
        self.entity_to_id = entity_to_id
        self.predicate_to_id = predicate_to_id
        self.lisp_style = lisp_style

    def map_to_id(self, s, p, o):
        return self.entity_to_id.get(str(s), s),\
               self.predicate_to_id.get(str(p), p),\
               self.entity_to_id.get(str(o), o)

    def toFQL(self, sq):
        """
        Convert SPARQL query to our custom language.

        :param sq: SPARQL query string
        :return: Functional query string
        """
        q = parseQuery(sq)
        g = Graph()

        askquery = True if q[1].name == 'AskQuery' else False
        countquery = False

        triples = q[1]['where']['part'][0]['triples']
        triples = [(i[0], i[1]['part'][0]['part'][0]['part'], i[2]) for i in triples]

        for triple in triples:
            g.add(triple)

        if askquery:
            # LC-QuAD has simple triple-based ASK queries; trivial to compute the output
            triple = triples[0]
            if self.lisp_style:
                out = "(in {} (find {} {}))".format(*self.map_to_id(triple[0], triple[2], triple[1]))
            else:
                out = "in(<{}>, find(<{}>, <{}>))".format(*self.map_to_id(triple[0], triple[2], triple[1]))
        else:
            try:
                # If it's not a COUNT query, then it's a SELECT
                if isinstance(q[1]['projection'][0]['expr']['expr']['expr']['expr']['expr']['expr'],
                              rdflib.plugins.sparql.parserutils.CompValue):
                    select = q[1]['projection'][0]['evar']
                    countquery = True
            except:
                select = q[1]['projection'][0]['var']

            triples_without_select_var = copy.deepcopy(g)
            triples_without_select_var -= g.triples((Variable(select), None, None))
            triples_without_select_var -= g.triples((None, None, Variable(select)))

            non_select_entities_in_where = list(triples_without_select_var.subjects()) + list(
                triples_without_select_var.objects())
            x_finds = []

            # Handle graph patterns involving the blank node '?x' and other entities/variables (excluding SELECT'ed variable)
            if non_select_entities_in_where:
                # Populate x_finds with find() statements that will output into ?x
                for s, p, o in triples_without_select_var:
                    s, p, o = self.map_to_id(s, p, o)
                    if s == Variable('x'):
                        if self.lisp_style:
                            x_finds.append("(find {} {})".format(o, p))
                        else:
                            x_finds.append("find(<{}>, <{}>)".format(o, p))
                    if o == Variable('x'):
                        if self.lisp_style:
                            x_finds.append("(find {} (reverse {}))".format(s, p))
                        else:
                            x_finds.append("find(<{}>, reverse(<{}>))".format(s, p))

                assert len(x_finds) in [1, 2]

                # Generate funq string for ?x
                if len(x_finds) == 1:
                    x_finds = x_finds[0]
                else:
                    if self.lisp_style:
                        x_finds = reduce(lambda a, b: "(intersection {} {})".format(a, b), x_finds)
                    else:
                        x_finds = reduce(lambda a, b: "intersection({}, {})".format(a, b), x_finds)

            select_finds = []
            select_g = Graph()
            select_g += g.triples((Variable(select), None, None))
            select_g += g.triples((None, None, Variable(select)))

            # Iterate over all graph patterns involving SELECT'ed variable.
            for s, p, o in select_g:
                s, p, o = self.map_to_id(s, p, o)
                if self.lisp_style:
                    if Variable('x') not in [s, o]:
                        # If no ?x references in subject or object, we have a simple find() retrieval output
                        if s == Variable(select):
                            select_finds.append("(find {} {})".format(o, p))
                        if o == Variable(select):
                            select_finds.append("(find {} (reverse {}))".format(s, p))
                    else:
                        # If we have a blank node ?x, paste x_finds in its place
                        if s == Variable('x'):
                            select_finds.append("(find {} (reverse {}))".format(x_finds, p))
                        if o == Variable('x'):
                            select_finds.append("(find {} {})".format(x_finds, p))
                else:
                    if Variable('x') not in [s, o]:
                        # If no ?x references in subject or object, we have a simple find() retrieval output
                        if s == Variable(select):
                            select_finds.append("find(<{}>, <{}>)".format(o, p))
                        if o == Variable(select):
                            select_finds.append("find(<{}>, reverse(<{}>))".format(s, p))
                    else:
                        # If we have a blank node ?x, paste x_finds in its place
                        if s == Variable('x'):
                            select_finds.append("find({}, reverse(<{}>))".format(x_finds, p))
                        if o == Variable('x'):
                            select_finds.append("find({}, <{}>)".format(x_finds, p))

            # Generate funq string for SELECT'ed variable
            if len(select_finds) == 1:
                select_finds = select_finds[0]
            else:
                if self.lisp_style:
                    select_finds = reduce(lambda a, b: "(intersection {} {})".format(a, b), select_finds)
                else:
                    select_finds = reduce(lambda a, b: "and({}, {})".format(a, b), select_finds)

            # If it's a COUNT query, just call a count() in the funq
            if countquery:
                if self.lisp_style:
                    select_finds = "(count {})".format(select_finds)
                else:
                    select_finds = "count({})".format(select_finds)

            out = select_finds

        return out

def generateFromOriginalLCQuAD(infile: str, converter: LCQuADSparqlToFunQuery):
    print(f"Parsing {infile}")
    with codecs.open(infile) as fp:
        data = json.load(fp)
        dataset = []
        for doc in tqdm(data):
            q = doc['corrected_question']
            sq = doc['sparql_query'].replace("COUNT(?uri)", "(COUNT(?uri) as ?uri)")
            g = converter.toFQL(sq)
            dataset.append({'question': q, 'logical_form': g})
    return dataset

def generateFromAnnotatedLCQuAD(infile: str, converter: LCQuADSparqlToFunQuery):
    print(f"Parsing {infile}")
    with codecs.open(infile) as fp:
        data = json.load(fp)
        dataset = []
        for doc in tqdm(data):
            q: str = doc['question']

            q_entity_replaced: str = q
            entities = []
            placeholder_count = 1

            for entity in doc['entity mapping']:
                mention = entity['label']
                uri = converter.entity_to_id[entity['uri']]

                start = q.find(mention)
                end = start + len(mention)
                location = (start, end)

                placeholder = f"ENT_{placeholder_count}"
                placeholder_count += 1

                q_entity_replaced = q[:start] + placeholder + q[end:]

                entities.append({'mention': mention,
                                 'uri': uri,
                                 'location': location,
                                 'placeholder': placeholder})

            sq = doc['sparql_query'].replace("COUNT(?uri)", "(COUNT(?uri) as ?uri)")
            g = converter.toFQL(sq)

            dataset.append({'question': q,
                            'question_mapped': q_entity_replaced,
                            'logical_form': g,
                            'entities': entities})
    return dataset

if __name__ == '__main__':
    dir_prefix = '/data/nilesh/datasets/LC-QuAD/'

    print("Loading DBpedia ID indexes")
    with codecs.open("/data/nilesh/datasets/dbpedia/triples.index.pickle.new", "rb") as f:
        (entity_to_id, predicate_to_id) = pickle.load(f)

    for key, value in entity_to_id.items():
        entity_to_id[key] = f"E{value}"

    for key, value in predicate_to_id.items():
        predicate_to_id[key] = f"P{value}"

    for lisp_style in [True, False]:
        for infile, outfile in [("annotated-test-data.json", "lcquad.annotated.{}.test.json"),
                                ("annotated-train-data.json", "lcquad.annotated.{}.train.json")]:
            outfile = outfile.format('lisp' if lisp_style else 'funq')

            converter = LCQuADSparqlToFunQuery(entity_to_id, predicate_to_id, lisp_style)

            dataset = generateFromAnnotatedLCQuAD(os.path.join(dir_prefix, infile), converter)

            with codecs.open(os.path.join(dir_prefix, outfile), "w") as fp:
                json.dump(dataset, fp, indent=4, separators=(',', ': '))