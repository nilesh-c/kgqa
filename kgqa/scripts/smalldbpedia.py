import codecs
from dataclasses import dataclass, field
from typing import Dict
import pickle
from tqdm import tqdm
import rdflib
from rdflib import Graph
from itertools import zip_longest
import re


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@dataclass
class Container:
    container: Dict[str, int] = field(default_factory=dict)
    counter: int = 0

    def get(self, element):
        if element in self.container:
            ret = self.container[element]
        else:
            ret = self.container[element] = self.counter
            self.counter += 1
        return ret


entities = Container()
predicates = Container()
triples = []

quoteLines = re.compile('<http://dbpedia.org/class/yago/.*".*>')


def fixQuotes(line):
    if '"' in line:
        quotesFound = list(quoteLines.finditer(line))
        if quotesFound:
            m = quotesFound[0]
            line = line[:m.start()] + line[m.start():m.end()].replace('"', '') + line[m.end():]
    return line

def main():
    """
    Reads all DBpedia triples and generates sequential IDs for entities and predicates
    """
    with codecs.open("alltriples.ttl") as ttl:
        ttl = tqdm(ttl)

        for buffer in grouper(ttl, 100000, ''):
            g = Graph()

            g.parse(data="".join(fixQuotes(i) for i in buffer), format="nt")

            for s, p, o in g:
                try:
                    if isinstance(o, rdflib.term.Literal):
                        continue

                    s = entities.get(str(s))
                    p = predicates.get(str(p))
                    o = entities.get(str(o))

                    triples.append((s, p, o))
                except Exception as e:
                    print(e)
                    print("ERROR: {}".format(s + p + o))

    with codecs.open("./dbpedia/triples.index.pickle", "wb") as f:
        pickle.dump((entities.container, predicates.container), f)

    with codecs.open("./dbpedia/triples.pickle", "wb") as f:
        pickle.dump(triples, f)

if __name__ == '__main__':
    main()