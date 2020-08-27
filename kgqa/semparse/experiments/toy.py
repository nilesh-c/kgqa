import json
import codecs
import torch
import random
import numpy as np

from allennlp.common import Params
from pathlib import Path
import faulthandler

from allennlp.models import SimpleSeq2Seq
from allennlp.predictors import SimpleSeq2SeqPredictor, Seq2SeqPredictor
from allennlp.training.tensorboard_writer import TensorboardWriter
from hdt import HDTDocument

from kgqa.data.lcquad_reader_simple import LCQuADReaderSimple
from kgqa.data.toy_reader import ToyReader
from kgqa.semparse.executor.stub_executor import StubExecutor
from kgqa.semparse.executor.hdt_executor import HdtExecutor
from allennlp.training import util as training_util

from allennlp.data import Vocabulary, DataIterator
from allennlp.data.iterators import BucketIterator
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.attention import DotProductAttention, BilinearAttention
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


class Toy():
    PROJECT_ROOT = (Path(__file__).parent / ".." / ".." / "..").resolve()  # pylint: disable=no-member
    MODULE_ROOT = PROJECT_ROOT / "kgqa"
    TEST_DATA_ROOT = MODULE_ROOT / "tests" / "data"

    def setUp(self):
        self.reader = ToyReader()
        self.train_instances = self.reader.read("/home/IAIS/nchakrabor/nmt_data/toy_reverse/train/toy_train.txt")
        self.dev_instances = self.reader.read("/home/IAIS/nchakrabor/nmt_data/toy_reverse/dev/toy_dev.txt")
        self.vocab = Vocabulary.from_instances(self.train_instances + self.dev_instances)

        token_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size('tokens') + 2,
                                    embedding_dim=256, padding_index=0)

        word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        encoder = PytorchSeq2SeqWrapper(nn.LSTM(input_size=word_embeddings.get_output_dim(),
                                                num_layers=2,
                                                hidden_size=256,
                                                bidirectional=True,
                                                dropout=0.4,
                                                batch_first=True))

        # self.set_up_model(model_params_file_path, dataset_sample_file_path)
        self.model = SimpleSeq2Seq(vocab=self.vocab,
                                   source_embedder=word_embeddings,
                                   encoder=encoder,
                                   target_embedding_dim=256,
                                   target_namespace='target_tokens',
                                   attention=DotProductAttention(),
                                   max_decoding_steps=25,
                                   beam_size=5,
                                   use_bleu=True
                                   )

        self.model.cuda(0)

        # iterator = dataiterator()
        # batch = next(iterator(self.instances, shuffle=false))
        # self.check_model_computes_gradients_correctly(self.model, batch)


    def test_model_training(self):
        serialization_dir = self.TEST_DATA_ROOT / "serialized_sample"
        tensorboard_dir = self.TEST_DATA_ROOT / "tensorboard.seq2seq"

        batch_size = 64

        train_iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], padding_noise=0.0, batch_size=batch_size)
        train_iterator.index_with(vocab=self.vocab)

        tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: np.ceil(len(self.train_instances) / batch_size),
            serialization_dir=tensorboard_dir,
            summary_interval=5,
            histogram_interval=5,
            should_log_parameter_statistics=True)

        trainer = CallbackTrainer(model=self.model,
                                  serialization_dir=serialization_dir,
                                  iterator=train_iterator,
                                  training_data=self.train_instances,
                                  num_epochs=1,
                                  cuda_device=0,
                                  optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3),
                                  callbacks=[LogToTensorboard(tensorboard),
                                             Validate(validation_data=self.dev_instances, validation_iterator=train_iterator),
                                             TrackMetrics(), ResetMetricsCallback()]
                                  )

        for i in range(50):
            print('Epoch: {}'.format(i))
            trainer.train()

            import itertools

            predictor = Seq2SeqPredictor(self.model, self.reader)

            for instance in itertools.islice(self.dev_instances, 10):
                print('SOURCE:', instance.fields['source_tokens'].tokens)
                print('GOLD:', instance.fields['target_tokens'].tokens)
                print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])


if __name__ == "__main__":
    random.seed(111)
    np.random.seed(333)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    run = Toy()
    run.setUp()
    # run.test_model_training()

        # for inst in self.instances:
        #     self.model(**inst)
        #     break

    # def test_model_can_train_save_and_load(self):
    #     self.ensure_model_can_train_save_and_load(self.param_file)
