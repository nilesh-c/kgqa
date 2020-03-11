import json
import codecs
import torch
import random
import numpy as np

from allennlp.common import Params
from pathlib import Path
import faulthandler

from allennlp.models import SimpleSeq2Seq
from allennlp.training.tensorboard_writer import TensorboardWriter
from hdt import HDTDocument

from kgqa.semparse.executor.stub_executor import StubExecutor
from kgqa.semparse.executor.hdt_executor import HdtExecutor
from allennlp.training import util as training_util

from allennlp.data import Vocabulary, DataIterator
from allennlp.data.iterators import BucketIterator
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.state_machines import BeamSearch
from allennlp.training import Trainer
from allennlp.training.callback_trainer import CallbackTrainer
from allennlp.training.callbacks import Validate, TrackMetrics, LogToTensorboard, Callback, handle_event, Events
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


class LCQuADMmlSemanticParsingRun():
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
        self.setUpExecutor()
        # self.setupstubexecutor()

        model_params_file_path = self.TEST_DATA_ROOT / "experiment.json"
        self.dataset_sample_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.v2.deurified.sample.json"
        self.dataset_train_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.v2.deurified.train.json"
        self.dataset_test_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.v2.deurified.test.json"
        predicates_file_path = self.TEST_DATA_ROOT / "properties.txt"
        with codecs.open(predicates_file_path) as fp:
            self.predicates = [i.strip() for i in fp]
            
        dbo_classes = set([dbo for dbo in self.predicates if dbo.split("/")[-1][0].isupper()])
        binary_predicates = set(self.predicates) - dbo_classes

        token_indexer = None #{'tokens': ELMoTokenCharactersIndexer()}

        if self.sample_only:
            sample_reader = LCQuADReader(executor=self.executor, predicates=binary_predicates, token_indexers=token_indexer, ontology_types=dbo_classes)
        else:
            train_reader = LCQuADReader(executor=self.executor, predicates=binary_predicates, token_indexers=token_indexer, ontology_types=dbo_classes)
            test_reader = LCQuADReader(executor=self.executor, predicates=binary_predicates, token_indexers=token_indexer, ontology_types=dbo_classes)

        # sample_reader.cache_data("sample_dataset")
        # train_reader.cache_data("train_dataset")
        # test_reader.cache_data("test_dataset")

        if self.sample_only:
            self.sample_instances = list(sample_reader.read(str(self.dataset_sample_file_path)))
        else:
            self.train_instances = list(train_reader.read(str(self.dataset_train_file_path)))
            self.test_instances = list(test_reader.read(str(self.dataset_test_file_path)))

        if self.sample_only:
            self.vocab = Vocabulary.from_instances(self.sample_instances)
        else:
            self.vocab = Vocabulary.from_instances(self.train_instances + self.test_instances)

        #self.vocab = Vocabulary()

        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size() + 2,
                                    embedding_dim=256, padding_index=0)

        #options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        #weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

        # the embedder maps the input tokens to the appropriate embedding matrix
        #elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
        #word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

        word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        encoder = PytorchSeq2SeqWrapper(nn.LSTM(input_size=word_embeddings.get_output_dim(),
                                                num_layers=1,
                                                hidden_size=128,
                                                bidirectional=True,
                                                # dropout=0.4,
                                                batch_first=True))

        val_outputs = self.TEST_DATA_ROOT / "val_outputs.json"

        self.val_outputs_fp = codecs.open(val_outputs, 'w')

        # self.set_up_model(model_params_file_path, dataset_sample_file_path)
        self.model = LCQuADMmlSemanticParser(vocab=self.vocab,
                                        sentence_embedder=word_embeddings,
                                        action_embedding_dim=256,
                                        encoder=encoder,
                                        attention=DotProductAttention(),
                                        decoder_beam_search=BeamSearch(beam_size=1),
                                        max_decoding_steps=50,
                                        dropout=0.5,
                                        val_outputs=self.val_outputs_fp)
        self.model.cuda(0)

        # iterator = dataiterator()
        # batch = next(iterator(self.instances, shuffle=false))
        # self.check_model_computes_gradients_correctly(self.model, batch)


    def test_model_forward(self):
        iterator = BucketIterator(sorting_keys=[("question", "num_tokens")], padding_noise=0.0, batch_size=5)
        iterator.index_with(vocab=self.vocab)
        batch = next(iterator(self.sample_instances, shuffle=False))
        self.check_model_computes_gradients_correctly(self.model, batch)

    def test_model_training(self):
        training_dataset = self.sample_instances if self.sample_only else self.train_instances
        #training_dataset = training_dataset[:500]
        validation_dataset = self.sample_instances if self.sample_only else self.test_instances
        serialization_dir = self.TEST_DATA_ROOT / "serialized_sample" if self.sample_only else "serialized"
        tensorboard_dir = self.TEST_DATA_ROOT / "tensorboard"

        batch_size = 64

        train_iterator = BucketIterator(sorting_keys=[("question", "num_tokens")], padding_noise=0.0, batch_size=batch_size)
        val_iterator = BucketIterator(sorting_keys=[("question", "num_tokens")], padding_noise=0.0, batch_size=batch_size)
        train_iterator.index_with(vocab=self.vocab)
        val_iterator.index_with(vocab=self.vocab)

        tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: np.ceil(len(training_dataset) / batch_size),
            serialization_dir=tensorboard_dir,
            summary_interval=5,
            histogram_interval=5,
            should_log_parameter_statistics=True)

        trainer = CallbackTrainer(model=self.model,
                                  serialization_dir=serialization_dir,
                                  iterator=train_iterator,
                                  training_data=training_dataset,
                                  num_epochs=20,
                                  cuda_device=0,
                                  optimizer=torch.optim.Adagrad(self.model.parameters()),
                                  callbacks=[LogToTensorboard(tensorboard),
                                             Validate(validation_data=validation_dataset, validation_iterator=val_iterator),
                                             TrackMetrics(), ResetMetricsCallback()]
                                  )

        trainer.train()

        self.val_outputs_fp.close()

if __name__ == "__main__":
    random.seed(111)
    torch.seed()
    np.random.seed(333)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    run = LCQuADMmlSemanticParsingRun()
    run.setUp()
    #run.test_model_training()

        # for inst in self.instances:
        #     self.model(**inst)
        #     break

    # def test_model_can_train_save_and_load(self):
    #     self.ensure_model_can_train_save_and_load(self.param_file)
