import json
import codecs
import torch

from allennlp.common import Params
from pathlib import Path
import faulthandler

from allennlp.models import SimpleSeq2Seq
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
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from kgqa.common.testcase import KGQATestCase
from kgqa.data import LCQuADReader
from kgqa.semparse.model import LCQuADMmlSemanticParser
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from torch import nn


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
        self.dataset_sample_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.sample.json"
        self.dataset_train_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.train.json"
        self.dataset_test_file_path = self.TEST_DATA_ROOT / "lcquad.annotated.lisp.test.json"
        predicates_file_path = self.TEST_DATA_ROOT / "properties.txt"
        with codecs.open(predicates_file_path) as fp:
            self.predicates = [i.strip() for i in fp]
            
        dbo_classes = set([dbo for dbo in self.predicates if dbo.split("/")[-1][0].isupper()])
        binary_predicates = set(self.predicates) - dbo_classes

        token_indexer = None #{'tokens': ELMoTokenCharactersIndexer()}

        if self.sample_only:
            sample_reader = LCQuADReader(executor=self.executor, predicates=binary_predicates, token_indexers=token_indexer)
        else:
            train_reader = LCQuADReader(executor=self.executor, predicates=binary_predicates, token_indexers=token_indexer)
            test_reader = LCQuADReader(executor=self.executor, predicates=binary_predicates, token_indexers=token_indexer)

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
                                    embedding_dim=128, padding_index=0)

        #options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
        #weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

        # the embedder maps the input tokens to the appropriate embedding matrix
        #elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
        #word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

        word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        encoder = PytorchSeq2SeqWrapper(nn.LSTM(input_size=word_embeddings.get_output_dim(),
                                                num_layers=2,
                                                hidden_size=512,
                                                bidirectional=True,
                                                dropout=0.1,
                                                batch_first=True))

        val_outputs = self.TEST_DATA_ROOT / "val_outputs.json"

        self.val_outputs_fp = codecs.open(val_outputs, 'w')

        # self.set_up_model(model_params_file_path, dataset_sample_file_path)
        self.model = LCQuADMmlSemanticParser(vocab=self.vocab,
                                        sentence_embedder=word_embeddings,
                                        action_embedding_dim=512,
                                        encoder=encoder,
                                        attention=DotProductAttention(),
                                        decoder_beam_search=BeamSearch(beam_size=5),
                                        max_decoding_steps=30,
                                        dropout=0.4,
                                        val_outputs=self.val_outputs_fp)
        # iterator = dataiterator()
        # batch = next(iterator(self.instances, shuffle=false))
        # self.check_model_computes_gradients_correctly(self.model, batch)


    def test_model_forward(self):
        iterator = BucketIterator(sorting_keys=[("question", "num_tokens")], padding_noise=0.0, batch_size=5)
        iterator.index_with(vocab=self.vocab)
        batch = next(iterator(self.sample_instances, shuffle=False))
        self.check_model_computes_gradients_correctly(self.model, batch)

    def test_model_training(self):
        train_iterator = BucketIterator(sorting_keys=[("question", "num_tokens")], padding_noise=0.0, batch_size=64)
        val_iterator = BucketIterator(sorting_keys=[("question", "num_tokens")], padding_noise=0.0, batch_size=64)
        train_iterator.index_with(vocab=self.vocab)
        val_iterator.index_with(vocab=self.vocab)

        trainer_params = Params({"num_epochs": 20,
                                "patience": 2,
                                "cuda_device": 0,
                                "optimizer": {
                                  "type": "adam"
                                }
                              })
        trainer = Trainer.from_params(model=self.model,
                                      serialization_dir=self.TEST_DATA_ROOT / "serialized_sample" if self.sample_only else "serialized",
                                      iterator=train_iterator,
                                      train_data=self.sample_instances if self.sample_only else self.train_instances,
                                      validation_data=self.sample_instances if self.sample_only else self.test_instances,
                                      params=trainer_params,
                                      validation_iterator=val_iterator)

        trainer.train()

        self.val_outputs_fp.close()

        if trainer._validation_data is not None:
            with torch.no_grad():
                val_loss, num_batches = trainer._validation_loss()
                val_metrics = training_util.get_metrics(trainer.model, val_loss, num_batches, reset=True)
                this_epoch_val_metric = val_metrics[trainer._validation_metric]
        else:
            val_metrics, this_epoch_val_metric = {}, {}
        print(val_metrics)

if __name__ == "__main__":
    faulthandler.enable()
    run = LCQuADMmlSemanticParsingRun()
    run.setUp()
    run.test_model_training()

        # for inst in self.instances:
        #     self.model(**inst)
        #     break

    # def test_model_can_train_save_and_load(self):
    #     self.ensure_model_can_train_save_and_load(self.param_file)
