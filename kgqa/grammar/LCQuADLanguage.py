from allennlp.semparse import DomainLanguage
from allennlp.semparse import predicate
from typing import NamedTuple, Set
from numbers import Number


class Predicate(str):
    pass

class Entity(str):
    pass

class LCQuADLanguage(DomainLanguage):
    """
    Implements the functions in our custom variable-free functional query language for the LCQuAD dataset.

    """

    def __init__(self, max_entity_id, max_predicate_id):
        super().__init__(start_types={Number, Entity, Predicate})
        eids = [f"E{x}" for x in range(max_entity_id)]
        pids = [f"P{x}" for x in range(max_predicate_id)]

        for i in eids:
            self.add_constant(i, Entity(i), type_=Entity)
        for i in pids:
            self.add_constant(i, Predicate(i), type_=Predicate)

    # @predicate
    # def find(self, entities: Set[Entity], predicate: Predicate) -> Set[Entity]:
    #     """
    #     Takes a set of entities E and a predicate p and loops through e in E and
    #     returns the set of entities that have an outgoing edge p to e.
    #
    #     """
    #     pass

    @predicate
    def find(self, entity: Entity, predicate: Predicate) -> Set[Entity]:
        """
        Takes an entity e and a predicate p and returns the set of entities that
        have an outgoing edge p to e.

        """
        pass

    @predicate
    def intersection(self, entities1: Set[Entity], entities2: Set[Entity]) -> Set[Entity]:
        """
        Return intersection of two sets of entities.
        """
        pass

    @predicate
    def get(self, entity: Entity) -> Set[Entity]:
        """
        Get entity and wrap it in a set.
        """
        pass

    @predicate
    def reverse(self, predicate: Predicate) -> Predicate:
        """
        Return the reverse of given predicate.
        """
        pass

    @predicate
    def count(self, entities: Set[Entity]) -> Number:
        """
        Returns a count of a set of entities.
        """
        pass


# if __name__ == '__main__':
#     l = LCQuADLanguage(17160596, 1922)