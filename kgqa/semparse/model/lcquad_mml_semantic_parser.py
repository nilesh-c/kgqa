import logging
from typing import Any, List, Dict, Union, Set
import numpy as np
import torch
import json
from collections import OrderedDict
from overrides import overrides
from joblib import Parallel, delayed

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import Activation
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.transition_functions import BasicTransitionFunction
from allennlp.models.semantic_parsing.wikitables.wikitables_mml_semantic_parser import WikiTablesMmlSemanticParser
from kgqa.semparse.language.lcquad_language import Entity
from kgqa.semparse.model.kg_embedders import KgEmbedder
from kgqa.semparse.model.lcquad_semantic_parser import LCQuADSemanticParser, OVERALL_SCORE, OVERALL_ACC_SCORE
from kgqa.semparse.language import LCQuADLanguage
from kgqa.training.trainers.mml import MaximumMarginalLikelihoodWithEval

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
from allennlp.models.semantic_parsing.atis.atis_semantic_parser import AtisSemanticParser

@Model.register("lcquad_mml_parser")
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
    training_beam_size : ``int``, optional (default=None)
        If given, we will use a constrained beam search of this size during training, so that we
        use only the top ``training_beam_size`` action sequences according to the model in the MML
        computation.  If this is ``None``, we will use all of the provided action sequences in the
        MML computation.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 decoder_beam_search: BeamSearch,
                 val_outputs,
                 max_decoding_steps: int,
                 training_beam_size: int = None,
                 dropout: float = 0.0) -> None:

        super().__init__(vocab=vocab,
                         sentence_embedder=sentence_embedder,
                         action_embedding_dim=action_embedding_dim,
                         encoder=encoder,
                         dropout=dropout)
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)
        self._decoder_step = BasicTransitionFunction(encoder_output_dim=self._encoder.get_output_dim(),
                                                     action_embedding_dim=action_embedding_dim,
                                                     input_attention=attention,
                                                     activation=Activation.by_name('relu')(),
                                                     add_action_bias=False,
                                                     dropout=dropout)
        self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self._action_padding_index = -1
        self.val_outputs = val_outputs

    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                question_predicates,
                # labelled_results,
                world: List[LCQuADLanguage],
                actions: List[List[ProductionRule]],
                question_entities=None,
                target_action_sequences: torch.LongTensor = None,
                labels: torch.LongTensor = None,
                logical_forms = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing type constrained target sequences, trained to maximize marginal
        likelihood over a set of approximate logical forms.
        """
        assert target_action_sequences is not None

        batch_size = question['tokens'].size()[0]
        # Remove the trailing dimension (from ListField[ListField[IndexField]]).
        # assert target_action_sequences.dim() == 3
        target_action_sequences = target_action_sequences.squeeze(-1)
        target_mask = target_action_sequences != self._action_padding_index

        # if self._kg_embedder:
        #     embedded_entities = self._kg_embedder(question_entities, input_type="entity")
        #     embedded_type_entities = self._kg_embedder(question_type_entities, input_type="entity")
        #     embedded_predicates = self._kg_embedder(question_predicates, input_type="predicate")

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

            # self._update_metrics(action_strings=batch_action_strings,
            #                     worlds=world,
            #                     labelled_results=labelled_results)

            debug_infos = []
            for i in range(batch_size):
                debug_infos.append(best_final_states[i][0].debug_info[0])

            action_mapping = {}
            for batch_index, batch_actions in enumerate(actions):
                for action_index, action in enumerate(batch_actions):
                    action_mapping[(batch_index, action_index)] = action[0]

            self._update_seq_metrics(action_strings=batch_action_strings,
                                     worlds=world,
                                     gold_logical_forms=logical_forms,
                                     train=self.training)

            outputs["predicted queries"] = batch_action_strings

            best_actions = batch_action_strings
            batch_action_info = []
            for batch_index, (predicted_actions, debug_info) in enumerate(zip(best_actions, debug_infos)):
                instance_action_info = []
                for predicted_action, action_debug_info in zip(predicted_actions[0], debug_info):
                    action_info = {}
                    action_info['predicted_action'] = predicted_action
                    considered_actions = action_debug_info['considered_actions']
                    probabilities = action_debug_info['probabilities']
                    actions = []
                    for action, probability in zip(considered_actions, probabilities):
                        if action != -1:
                            actions.append((action_mapping[(batch_index, action)], probability))
                    actions.sort()
                    considered_actions, probabilities = zip(*actions)
                    action_info['considered_actions'] = considered_actions
                    action_info['action_probabilities'] = probabilities
                    action_info['question_attention'] = action_debug_info.get('question_attention', [])
                    instance_action_info.append(action_info)
                batch_action_info.append(instance_action_info)
            outputs["predicted_actions"] = batch_action_info

        return outputs

    def _update_seq_metrics(self,
                            action_strings: List[List[List[str]]],
                            worlds: List[LCQuADLanguage],
                            gold_logical_forms: List[str],
                            train=False):
        batch_size = len(worlds)
        for i in range(batch_size):
            generated_lf = worlds[i].action_sequence_to_logical_form(action_strings[i][0]) if action_strings else ''
            gold_lf = gold_logical_forms[i][0]
            if not train:
                self.val_outputs.write(json.dumps({"gold": gold_lf, "generated": generated_lf}))
                self.val_outputs.write("\n")
                self.val_outputs.write("\n")
                self.val_outputs.flush()

            self._compute_instance_seq_metric(generated_lf, gold_lf)

        for metric_name, metric in self._metrics.items():
            overall_accuracy_metric = self._metrics[OVERALL_ACC_SCORE]
            if metric_name == "accuracy":
                accuracy = metric.get_metric()
                overall_accuracy_metric(accuracy)


    # noinspection PyTypeChecker
    def _update_metrics(self,
                        action_strings: List[List[List[str]]],
                        worlds: List[LCQuADLanguage],
                        labelled_results: List[Union[bool, int, Set[Entity]]]) -> None:

        batch_size = len(worlds)

        def retrieve_results(action_strings, world):
            action_sequence = action_strings[0] if action_strings else []
            return world.execute_action_sequence(action_sequence)

        # retrieved_results = Parallel(n_jobs=10)(delayed(retrieve_results)(actions, world) for actions, world
        #                                                 in zip(action_strings[:batch_size], worlds[:batch_size]))
        # retrieved_results = list(map(retrieve_results, action_strings[:batch_size], worlds[:batch_size]))

        for i in range(batch_size):
            self._compute_instance_metrics(action_strings[i][0] if action_strings else [],
                                           labelled_results[i], worlds[i])

        # Update overall score.
        for metric_name, metric in self._metrics.items():
            overall_score_metric = self._metrics[OVERALL_SCORE]
            overall_accuracy_metric = self._metrics[OVERALL_ACC_SCORE]
            if metric_name == "accuracy":
                accuracy = metric.get_metric()
                # if metric_name.replace("accuracy", "") not in self.retrieval_question_types:
                #     overall_score_metric(accuracy)
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
            metrics[key] = metric.get_metric(reset) # if not self.training else -1
        return metrics
