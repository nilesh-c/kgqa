import logging
from typing import Dict

from overrides import overrides
import numpy as np
import pickle
import torch
from torch.nn.functional import embedding

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.nn import util
from allennlp.modules.time_distributed import TimeDistributed

from dialogue_models.modules.kg_embedders.kg_embedder import KgEmbedder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@KgEmbedder.register("kg_embedding")
class KgEmbedding(KgEmbedder):
    """
    A simple KG embedder with separate embedding vocabs and matrices for entities and embeddings. It
    uses separate projection matrices for entities and predicates if a ``projection_dim`` is provided,
    which only makes sense when ``trainable`` is ``False``. If an id is not present, a new embedding
    is added to the embedding matrix on the fly.

    # TODO check if embedding matrices extended with missing ids is saved after training.

    Parameters
    ----------
    num_entities : int
        Size of the dictionary of embeddings (vocabulary size) of entities.
    num_predicates : int
        Size of the dictionary of embeddings (vocabulary size) of predicates.
    embedding_dim : int
        The size of each embedding vector (both entities and predicates).
    projection_dim : int, (optional, default=None)
        If given, add a projection layer after the embedding layer.  This really only makes
        sense if ``trainable`` is ``False``.
    entity_weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the entity embedding lookup, allowing the use of
        pretrained vectors.
    predicate_weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the predicate embedding lookup, allowing the use of
        pretrained vectors.
    entity2id : Dict[str, int]
        Vocab that maps entities to the corresponding index of the embedding matrix.
    predicate2id : Dict[str, int]
        Vocab that maps predicates to the corresponding index of the embedding matrix.
    padding_index : int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : bool, (optional, default=True)
        Whether or not to optimize the embedding parameters.
    max_norm : float, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : float, (optional, default=2)
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : boolean, (optional, default=False)
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : bool, (optional, default=False)
        Whether or not the PyTorch backend should use a sparse representation of the embedding weight.

    Returns
    -------
    An Embedding module.
    """
    def __init__(self,
                 num_entities: int,
                 num_predicates: int,
                 embedding_dim: int,
                 projection_dim: int = None,
                 entity_weight: torch.FloatTensor = None,
                 predicate_weight: torch.FloatTensor = None,
                 entity2id: Dict[str, int] = None,
                 predicate2id: Dict[str, int] = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 cuda_device: int = -1) -> None:
        super(KgEmbedding, self).__init__()
        self.num_entities = num_entities
        self.num_predicates = num_predicates
        self.embedding_dim = embedding_dim
        self.entity2id = entity2id
        self.predicate2id = predicate2id
        self.padding_index = padding_index
        self.trainable = trainable
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.cuda_device = cuda_device
        self._projection_entity = None
        self._projection_predicate = None

        # Init weights if not provided.
        self.entity_weight = self.init_weight(entity_weight, num_entities, embedding_dim, trainable)
        self.predicate_weight = self.init_weight(predicate_weight, num_predicates, embedding_dim, trainable)

        # Add padding.
        if self.padding_index is not None:
            self.entity_weight.data[self.padding_index].fill_(0)
            self.predicate_weight.data[self.padding_index].fill_(0)

        # Init projection weights if used.
        self.output_dim = projection_dim or embedding_dim
        if projection_dim:
            self._projection_entity = torch.nn.Linear(embedding_dim, projection_dim)
            self._projection_predicate = torch.nn.Linear(embedding_dim, projection_dim)

    @staticmethod
    def init_weight(weight, num_elements, embedding_dim, trainable):
        if weight is None:
            weight = torch.FloatTensor(num_elements, embedding_dim)
            weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(weight)
        else:
            if weight.size() != (num_elements, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
        return torch.nn.Parameter(weight, requires_grad=trainable)

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, inputs, input_type='entity'):  # pylint: disable=arguments-differ
        # Use to specified input type.
        if input_type == 'entity':
            weight = self.entity_weight
            element2id = self.entity2id
        elif input_type == 'predicate':
            weight = self.predicate_weight
            element2id = self.predicate2id
        else:
            raise Exception("{} is not a valid input type, use 'entity' or 'predicate'.".format(x))

        # Find ids and add new ones if non-existent.
        max_len = max([len(input) for input in inputs])
        for i, input in enumerate(inputs):
            ids = []
            for key in input:
                if key not in element2id:
                    element2id[key] = len(weight)
                    weight = self.add_new_embedding(weight, input_type)
                ids.append(element2id[key])
            inputs[i] = ids + [0] * (max_len - len(input))
        inputs = torch.LongTensor(inputs)

        # Find embeddings of ids.
        original_size = inputs.size()
        inputs = util.combine_initial_dims(inputs)
        inputs = util.move_to_device(inputs, self.cuda_device)
        embedded = embedding(inputs, weight,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)
        embedded = util.uncombine_initial_dims(embedded, original_size)
        return self.project(embedded, input_type)

    def add_new_embedding(self, weight, input_type):
        new_embedding = torch.FloatTensor(1, self.embedding_dim)
        new_embedding = torch.nn.Parameter(new_embedding)
        torch.nn.init.xavier_uniform_(new_embedding)
        weight = torch.cat((weight, new_embedding), 0)
        if input_type == 'entity':
            self.entity_weight = torch.nn.Parameter(weight, requires_grad=self.trainable)
        elif input_type == 'predicate':
            self.predicate_weight = torch.nn.Parameter(weight, requires_grad=self.trainable)
        return weight

    @classmethod
    def from_params(cls, params: Params) -> 'Embedding':
        # pylint: disable=arguments-differ
        num_entities = params.pop_int('num_entities', None)
        num_predicates = params.pop_int('num_predicates', None)
        embedding_dim = params.pop_int('embedding_dim')
        entity_pretrained_file = params.pop("entity_pretrained_file", None)
        predicate_pretrained_file = params.pop("predicate_pretrained_file", None)
        entity2id_file = params.pop('entity2id_file', None)
        predicate2id_file = params.pop('predicate2id_file', None)
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        padding_index = params.pop_int('padding_index', None)
        max_norm = params.pop_float('max_norm', None)
        norm_type = params.pop_float('norm_type', 2.)
        scale_grad_by_freq = params.pop_bool('scale_grad_by_freq', False)
        sparse = params.pop_bool('sparse', False)
        cuda_device = params.pop_int('cuda_device', -1)
        params.assert_empty(cls.__name__)
        entity_weight, predicate_weight = None, None

        entity2id = _read_element2id_file(entity2id_file)
        predicate2id = _read_element2id_file(predicate2id_file)

        if entity_pretrained_file:
            entity_weight = _read_pretrained_embeddings_file(entity_pretrained_file)

        if predicate_pretrained_file:
            predicate_weight = _read_pretrained_embeddings_file(predicate_pretrained_file)

        return cls(num_entities=num_entities,
                   num_predicates=num_predicates,
                   embedding_dim=embedding_dim,
                   projection_dim=projection_dim,
                   entity2id=entity2id,
                   predicate2id=predicate2id,
                   entity_weight=entity_weight,
                   predicate_weight=predicate_weight,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse,
                   cuda_device=cuda_device)

    def project(self, embedded, input_type):
        if input_type == 'entity':
            if self._projection_entity:
                projection = self._projection_entity
                for _ in range(embedded.dim() - 2):
                    projection = TimeDistributed(projection)
                embedded = projection(embedded)
        elif input_type == 'predicate':
            if self._projection_predicate:
                projection = self._projection_predicate
                for _ in range(embedded.dim() - 2):
                    projection = TimeDistributed(projection)
                embedded = projection(embedded)
        return embedded


def _read_pretrained_embeddings_file(path: str) -> torch.FloatTensor:
    path = cached_path(path)
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    return torch.FloatTensor(embeddings)


def _read_element2id_file(element2id_file):
    element2id_file = cached_path(element2id_file)
    element2id = {}
    with open(element2id_file, 'r') as f:
        next(f)
        for line in f:
            key_value = line.split('\t')
            element2id[key_value[0]] = int(key_value[1])
    return element2id
