import json
import codecs
from typing import Iterable

import torch
import random
import numpy as np

from allennlp.common import Params
from pathlib import Path
import faulthandler

from allennlp.models import SimpleSeq2Seq
from allennlp.predictors import SimpleSeq2SeqPredictor, Seq2SeqPredictor
from allennlp.training.learning_rate_schedulers import LearningRateScheduler, CosineWithRestarts
from allennlp.training.tensorboard_writer import TensorboardWriter
from hdt import HDTDocument

from kgqa.data.lcquad_reader_simple import LCQuADReaderSimple
from kgqa.semparse.executor.stub_executor import StubExecutor
from kgqa.semparse.executor.hdt_executor import HdtExecutor
from allennlp.training import util as training_util

from allennlp.data import Vocabulary, DataIterator, Instance, DatasetReader
from allennlp.data.iterators import BucketIterator, MultiprocessIterator
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.attention import DotProductAttention, BilinearAttention
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.state_machines import BeamSearch
from allennlp.training import Trainer
from allennlp.training.callback_trainer import CallbackTrainer
from allennlp.training.callbacks import Validate, TrackMetrics, LogToTensorboard, Callback, handle_event, Events, \
    UpdateLearningRate
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from kgqa.common.testcase import KGQATestCase
from kgqa.data import LCQuADReader
from kgqa.semparse.model import LCQuADMmlSemanticParser
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from torch import nn

class ResetMetricsCallback(Callback):
    @handle_event(Events.VALIDATE, priority=-50)
    def reset_training_metrics(self, trainer):
        print("Resetting metrics", trainer.model.get_metrics(reset=True))

class ValidationLogCallback(Callback):
    def __init__(self, reader: DatasetReader, val_instances: Iterable[Instance]):
        self.reader = reader
        self.val_instances = val_instances

    @handle_event(Events.VALIDATE, priority=-50)
    def reset_training_metrics(self, trainer: CallbackTrainer):
        import itertools

        predictor = Seq2SeqPredictor(trainer.model, self.reader)

        print('Epoch', trainer.epoch_number)
        print('Val comparison')
        for instance in itertools.islice(self.val_instances, 5):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])

        print("Train comparison")
        for instance in itertools.islice(trainer.training_data, 5):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])

class LCQuADMmlSemanticParsingSimple():
    PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / "..").resolve()  # pylint: disable=no-member
    MODULE_ROOT = PROJECT_ROOT / "kgqa"
    TEST_DATA_ROOT = MODULE_ROOT / "tests" / "data"

    @classmethod
    def setUpExecutor(self):
        hdt = HDTDocument('/home/IAIS/nchakrabor/datasets/hdt/dbpedia2016-04en.hdt', map=True, progress=True)
        self.executor = HdtExecutor(graph=hdt)

    @classmethod
    def setUpStubExecutor(self):
        self.executor = StubExecutor()

    def setUp(self):
        self.sample_only = False
        # self.setupstubexecutor()

        model_params_file_path = self.TEST_DATA_ROOT / "experiment.json"
        self.dataset_sample_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.v3.deurified.simple.sample.json"
        self.dataset_train_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.v3.train.json"
        self.dataset_test_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.v3.test.json"
        predicates_file_path = self.TEST_DATA_ROOT / "properties.txt"
        with codecs.open(predicates_file_path) as fp:
            self.predicates = [i.strip() for i in fp]

        dbo_classes = set([dbo for dbo in self.predicates if dbo.split("/")[-1][0].isupper()])
        binary_predicates = set(self.predicates) - dbo_classes

        if self.sample_only:
            self.sample_reader = LCQuADReaderSimple(predicates=binary_predicates, ontology_types=dbo_classes)
        else:
            self.train_reader = LCQuADReaderSimple(predicates=binary_predicates, ontology_types=dbo_classes)
            # self.test_reader = LCQuADReaderSimple(predicates=binary_predicates, ontology_types=dbo_classes)



        # sample_reader.cache_data("sample_dataset")
        # train_reader.cache_data("train_dataset")
        # test_reader.cache_data("test_dataset")

        if self.sample_only:
            self.sample_instances = list(self.sample_reader.read(str(self.dataset_sample_file_path)))
        else:
            self.train_instances = list(self.train_reader.read(str(self.dataset_train_file_path)))
            self.test_instances = list(self.train_reader.read(str(self.dataset_test_file_path)))

        if self.sample_only:
            self.vocab = Vocabulary.from_instances(self.sample_instances)
        else:
            self.vocab = Vocabulary.from_instances(self.train_instances + self.test_instances, min_count={'tokens': 3, 'target_tokens': 3})
            #min_count={'tokens': 3, 'target_tokens': 3})

        #self.vocab = Vocabulary()

        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size('tokens') + 2,
                                    embedding_dim=512, padding_index=0)

        #options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        #weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

        # the embedder maps the input tokens to the appropriate embedding matrix
        #elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
        #word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

        word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        encoder = PytorchSeq2SeqWrapper(nn.LSTM(input_size=word_embeddings.get_output_dim(),
                                                num_layers=2,
                                                hidden_size=256,
                                                bidirectional=True,
                                                dropout=0.5,
                                                batch_first=True))

        val_outputs = self.TEST_DATA_ROOT / "val_outputs.seq2seq.json"

        self.val_outputs_fp = codecs.open(val_outputs, 'w')

        # self.set_up_model(model_params_file_path, dataset_sample_file_path)
        self.model = SimpleSeq2Seq(vocab=self.vocab,
                                   source_embedder=word_embeddings,
                                   encoder=encoder,
                                   target_embedding_dim=128,
                                   target_namespace='target_tokens',
                                   attention=DotProductAttention(),
                                   max_decoding_steps=25,
                                   beam_size=5,
                                   use_bleu=True,
                                   scheduled_sampling_ratio=0.3
                                   )

        self.model.cuda(0)

        # iterator = dataiterator()
        # batch = next(iterator(self.instances, shuffle=false))
        # self.check_model_computes_gradients_correctly(self.model, batch)


    def test_model_forward(self):
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], padding_noise=0.0, batch_size=5)
        iterator.index_with(vocab=self.vocab)
        batch = next(iterator(self.sample_instances, shuffle=False))
        self.check_model_computes_gradients_correctly(self.model, batch)

    def test_model_training(self):
        training_dataset = self.sample_instances if self.sample_only else self.train_instances
        #training_dataset = training_dataset[:500]
        validation_dataset = self.sample_instances if self.sample_only else self.test_instances
        serialization_dir = self.TEST_DATA_ROOT / "serialized_sample" if self.sample_only else "serialized"
        tensorboard_dir = self.TEST_DATA_ROOT / "tensorboard.seq2seq"

        batch_size = 64

        train_iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], padding_noise=0.1, batch_size=batch_size)
        train_iterator.index_with(vocab=self.vocab)
        multiproc_iterator = MultiprocessIterator(train_iterator, num_workers=4, output_queue_size=6000)

        tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: np.ceil(len(training_dataset) / batch_size),
            serialization_dir=tensorboard_dir,
            summary_interval=5,
            histogram_interval=5,
            should_log_parameter_statistics=True,
            should_log_learning_rate=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = CosineWithRestarts(optimizer=optimizer, t_initial=5)

        trainer = CallbackTrainer(model=self.model,
                                  serialization_dir=serialization_dir,
                                  iterator=multiproc_iterator,
                                  training_data=self.train_instances,
                                  num_epochs=100,
                                  cuda_device=0,
                                  optimizer=optimizer,
                                  callbacks=[LogToTensorboard(tensorboard),
                                             Validate(validation_data=self.test_instances,
                                                      validation_iterator=multiproc_iterator),
                                             TrackMetrics(),
                                             ResetMetricsCallback(),
                                             UpdateLearningRate(scheduler),
                                             ValidationLogCallback(self.train_reader, self.test_instances)]
                                  )

        # trainer = Trainer(model=self.model,
        #                   serialization_dir=serialization_dir,
        #                   iterator=train_iterator,
        #                   train_dataset=training_dataset,
        #                   num_epochs=1,
        #                   cuda_device=0,
        #                   optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3),
        #                   validation_dataset=training_dataset,
        #                   validation_iterator=train_iterator,
        #                   should_log_learning_rate=True,
        #                   learning_rate_scheduler=scheduler
        #                   )

        # for i in range(50):
        #     print('Epoch: {}'.format(i))
        #     trainer.train()
        #
        #     import itertools
        #
        #     predictor = Seq2SeqPredictor(self.model, self.train_reader)
        #
        #     for instance in itertools.islice(training_dataset, 10):
        #         print('SOURCE:', instance.fields['source_tokens'].tokens)
        #         print('GOLD:', instance.fields['target_tokens'].tokens)
        #         print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])
        #
        # self.val_outputs_fp.close()

        trainer.train()

def export_opennmt(run):
    language = run.train_reader.language
    productions = {prod:num for num, prod in enumerate(language.all_possible_productions())}

    for stage in ["train", "test"]:
        for xy in ["source", "target"]:
            filename = xy + "." + stage + '.simple.dataset'
            with codecs.open(filename, "w") as fp:
                for instance in getattr(run, stage + '_instances'):
                    a = instance.fields[xy + '_tokens']
                    if xy == 'source':
                        tokens = a.tokens[1:-1]
                    else:
                        tokens = [productions[i.text] for i in a.tokens[1:-1]]

                    fp.write(" ".join([str(i) for i in tokens]))
                    fp.write("\n")

    with codecs.open('productions.dict.json', "w") as fp:
        json.dump(productions, fp)

if __name__ == "__main__":
    random.seed(111)
    np.random.seed(333)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    run = LCQuADMmlSemanticParsingSimple()
    run.setUp()
    export_opennmt(run)
    # run.test_model_training()




        # for inst in self.instances:
        #     self.model(**inst)
        #     break

    # def test_model_can_train_save_and_load(self):
    #     self.ensure_model_can_train_save_and_load(self.param_file)
