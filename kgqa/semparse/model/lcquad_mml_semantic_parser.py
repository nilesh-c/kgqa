import logging
from typing import Any, List, Dict, Union, Set

import torch
from collections import OrderedDict
from overrides import overrides

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import Activation
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.transition_functions import BasicTransitionFunction

from kgqa.semparse.language.lcquad_language import Entity
from kgqa.semparse.model.kg_embedders import KgEmbedder
from kgqa.semparse.model.lcquad_semantic_parser import LCQuADSemanticParser, OVERALL_SCORE, OVERALL_ACC_SCORE
from kgqa.semparse.language import LCQuADLanguage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("csqa_mml_parser")
class LCQuADMmlSemanticParser(LCQuADSemanticParser):
    """
    ``LCQuADMmlSemanticParser`` is an ``LCQuADSemanticParser`` that solves the problem of lack of
    logical form annotations by maximizing the marginal likelihood of an approximate set of target
    sequences that yield the correct denotation. This parser takes the output of an offline search
    process as the set of target sequences for training, the latter performs search during training.

    Parameters
    ----------
    vocab : ``Vocabulary``
        Passed to super-class.
    sentence_embedder : ``TextFieldEmbedder``
        Passed to super-class.
    action_embedding_dim : ``int``
        Passed to super-class.
    encoder : ``Seq2SeqEncoder``
        Passed to super-class.
    attention : ``Attention``
        We compute an attention over the input question at each step of the decoder, using the
        decoder hidden state as the query.  Passed to the TransitionFunction.
    decoder_beam_search : ``BeamSearch``
        Beam search used to retrieve best sequences after training.
    max_decoding_steps : ``int``
        Maximum number of steps for beam search after training.
    dropout : ``float``, optional (default=0.0)
        Probability of dropout to apply on encoder outputs, decoder outputs and predicted actions.
    kg_embedder : ``KgEmbedder``
        Embedder for knowledge graph elements.
    direct_questions_only : ``bool``, optional (default=True)
        Only train on direct question (i.e.: without questions that refer to earlier conversation).
    """

    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 dropout: float = 0.0,
                 kg_embedder: KgEmbedder = None,
                 direct_questions_only=True) -> None:

        super().__init__(vocab=vocab,
                         sentence_embedder=sentence_embedder,
                         kg_embedder=kg_embedder,
                         action_embedding_dim=action_embedding_dim,
                         encoder=encoder,
                         dropout=dropout,
                         direct_questions_only=direct_questions_only)

        self._decoder_trainer = MaximumMarginalLikelihood()
        self._decoder_step = BasicTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                     action_embedding_dim=action_embedding_dim,
                                                     input_attention=attention,
                                                     num_start_types=1,
                                                     activation=Activation.by_name('tanh')(),
                                                     predict_start_type_separately=False,
                                                     add_action_bias=False,
                                                     dropout=dropout)
        self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1

    @overrides
    def forward(self,  # type: ignore
                qa_id,
                question: Dict[str, torch.LongTensor],
                question_type,  # TODO add types to arguments
                question_description,
                question_predicates,
                labelled_results,
                world: List[LCQuADLanguage],
                actions: List[List[ProductionRule]],
                question_entities=None,
                identifier: List[str] = None,
                question_segments: torch.FloatTensor = None,
                target_action_sequences: torch.LongTensor = None,
                labels: torch.LongTensor = None,
                logical_forms: str = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing type constrained target sequences, trained to maximize marginal
        likelihood over a set of approximate logical forms.
        """
        assert target_action_sequences is not None

        batch_size = question['tokens'].size()[0]
        # Remove the trailing dimension (from ListField[ListField[IndexField]]).
        target_action_sequences = target_action_sequences.squeeze(-1)
        target_mask = target_action_sequences != self._action_padding_index

        if self._kg_embedder:
            embedded_entities = self._kg_embedder(question_entities, input_type="entity")
            embedded_type_entities = self._kg_embedder(question_type_entities, input_type="entity")
            embedded_predicates = self._kg_embedder(question_predicates, input_type="predicate")

        initial_rnn_state = self._get_initial_rnn_state(question)
        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float) for _ in range(batch_size)]

        # TODO (pradeep): Assuming all worlds give the same set of valid actions.
        initial_grammar_statelet = [self._create_grammar_state(world[i], actions[i]) for i in range(batch_size)]
        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_statelet,
                                          possible_actions=actions)

        outputs = self._decoder_trainer.decode(initial_state,
                                               self._decoder_step,
                                               (target_action_sequences, target_mask))

        # print(outputs)

        if not self.training:
            initial_state.debug_info = [[] for _ in range(batch_size)]
            best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                                 initial_state,
                                                                 self._decoder_step,
                                                                 keep_final_unfinished_states=False)
            best_action_sequences: Dict[int, List[List[int]]] = {}
            for i in range(batch_size):
                # Decoding may not have terminated with any completed logical forms, if `num_steps`
                # isn't long enough (or if the model is not trained enough and gets into an
                # infinite action loop).
                if i in best_final_states:
                    best_action_indices = [best_final_states[i][0].action_history[0]]
                    best_action_sequences[i] = best_action_indices
            batch_action_strings = self._get_action_strings(actions, best_action_sequences)

            self._update_metrics(action_strings=batch_action_strings,
                                 worlds=world,
                                 labelled_results=labelled_results)

            outputs["predicted queries"] = batch_action_strings

        return outputs

    # noinspection PyTypeChecker
    def _update_metrics(self,
                        action_strings: List[List[List[str]]],
                        worlds: List[LCQuADLanguage],
                        labelled_results: List[Union[bool, int, Set[Entity]]]) -> None:

        batch_size = len(worlds)

        for i in range(batch_size):
            # Taking only the best sequence.
            instance_action_strings = action_strings[i][0] if action_strings[i] else []

            self._compute_instance_metrics(instance_action_strings, labelled_results[i], worlds[i])

        # Update overall score.
        for metric_name, metric in self._metrics.items():
            overall_score_metric = self._metrics[OVERALL_SCORE]
            overall_accuracy_metric = self._metrics[OVERALL_ACC_SCORE]
            if metric_name == "accuracy":
                accuracy = metric.get_metric()
                if metric_name.replace(" accuracy", "") not in self.retrieval_question_types:
                    overall_score_metric(accuracy)
                overall_accuracy_metric(accuracy)

            elif metric_name == "precision":
                precision = metric.get_metric()
                recall = self._metrics["recall"].get_metric()
                f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
                overall_score_metric(f1)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = OrderedDict()
        for key, metric in self._metrics.items():
            # tensorboard_key = key_to_tensorboard_key(key)
            # always write something (None) to dict to preserve metric ordering
            metrics[key] = metric.get_metric(reset) if not self.training else None
        return metrics
