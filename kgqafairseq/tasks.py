import os
import torch

from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.models.roberta import XLMRModel, RobertaHubInterface
from fairseq.tasks import FairseqTask, register_task
from kgqafairseq.src.data.dictionary import BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from kgqafairseq.src.utils import AttrDict

@register_task('semparse_seq2seq')
class SemparseSeq2SeqTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='DATA_PATH',
                            help='file prefix for data')
        parser.add_argument('--target-lang-dir',  default='', type=str,
                            help='file prefix for low-resource target language data')
        parser.add_argument('--max-input-length', default=50, type=int,
                            help='max input utterance length')
        parser.add_argument('--max-output-length', default=50, type=int,
                            help='max output logical form length')

    @classmethod
    def setup_task(cls, args, xlmr=None, **kwargs):

        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.
        if not xlmr:
            xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base.v0')
        input_vocab = xlmr.task.dictionary
        # xlmr = None
        # input_vocab = Dictionary.load(os.path.join(args.data, 'dict.input.txt'))
        output_vocab = Dictionary.load(os.path.join(args.data, 'dict.label.txt'))
        print('| [input] dictionary: {} types'.format(len(input_vocab)))
        print('| [label] dictionary: {} types'.format(len(output_vocab)))

        return SemparseSeq2SeqTask(args, input_vocab, output_vocab, xlmr)

    def __init__(self, args, input_vocab, output_vocab, xlmr: RobertaHubInterface):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.xlmr = xlmr

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        target_lang = split.startswith('target_')
        prefix = os.path.join(self.args.target_lang_dir if target_lang else self.args.data,
                              '{}.input-label'.format(split.replace("target_", "")))

        # Read input sentences.
        input_utterances, input_lengths = [], []
        with open(prefix + '.input', encoding='utf-8') as file:
            for line in file:
                input_utterance = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.xlmr.encode(input_utterance)
                # tokens = self.input_vocab.encode_line(
                #     input_utterance, add_if_not_exist=False,
                # ).to(torch.long)

                input_utterances.append(tokens)
                input_lengths.append(tokens.numel())

        # Read labels.
        output_lfs, output_lengths = [], []
        with open(prefix + '.label', encoding='utf-8') as file:
            for line in file:
                output_lf = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.output_vocab.encode_line(
                    output_lf, add_if_not_exist=False,
                ).to(torch.long)

                output_lfs.append(tokens)
                output_lengths.append(tokens.numel())

        print('| {} {} {} examples'.format(self.args.data, split, len(input_utterances)))

        # We reuse LanguagePairDataset since classification can be modeled as a
        # sequence-to-sequence task where the target sequence has length 1.
        self.datasets[split] = LanguagePairDataset(
            src=input_utterances,
            src_sizes=input_lengths,
            src_dict=self.input_vocab,
            tgt=output_lfs,
            tgt_sizes=output_lengths,  # targets have length 1
            tgt_dict=self.output_vocab,
            left_pad_source=False,
            max_source_positions=self.args.max_input_length,
            max_target_positions=self.args.max_output_length,
            input_feeding=True,
        )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return (self.args.max_input_length, self.args.max_output_length)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.output_vocab

@register_task('semparse_classification')
class SemparseClassificationTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='DATA_PATH',
                            help='file prefix for data')
        parser.add_argument('--max-input-length', default=50, type=int,
                            help='max input utterance length')

    @classmethod
    def setup_task(cls, args, encoder=None, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.
        xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base.v0')
        input_vocab = xlmr.task.dictionary
        # xlmr = None
        # input_vocab = Dictionary.load(os.path.join(args.data, 'dict.input.txt'))
        output_vocab = Dictionary.load(os.path.join(args.data, 'dict.label.txt'))
        print('| [input] dictionary: {} types'.format(len(input_vocab)))
        print('| [label] dictionary: {} types'.format(len(output_vocab)))

        return SemparseClassificationTask(args, input_vocab, output_vocab, xlmr)

    def __init__(self, args, input_vocab, output_vocab, xlmr: RobertaHubInterface):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.xlmr = xlmr

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        prefix = os.path.join(self.args.data, '{}.input-label'.format(split))

        # Read input sentences.
        input_utterances, input_lengths = [], []
        with open(prefix + '.input', encoding='utf-8') as file:
            for line in file:
                input_utterance = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.xlmr.encode(input_utterance)
                # tokens = self.input_vocab.encode_line(
                #     input_utterance, add_if_not_exist=False,
                # ).to(torch.long)

                input_utterances.append(tokens)
                input_lengths.append(tokens.numel())

        # Read labels.
        labels = []
        with open(prefix + '.label', encoding='utf-8') as file:
            for line in file:
                label = line.strip()
                labels.append(
                    # Convert label to a numeric ID.
                    torch.LongTensor({self.output_vocab.add_symbol(label)})
                )

        assert len(input_utterances) == len(labels)
        print('| {} {} {} examples'.format(self.args.data, split, len(input_utterances)))

        # We reuse LanguagePairDataset since classification can be modeled as a
        # sequence-to-sequence task where the target sequence has length 1.
        self.datasets[split] = LanguagePairDataset(
            src=input_utterances,
            src_sizes=input_lengths,
            src_dict=self.input_vocab,
            tgt=labels,
            tgt_sizes=torch.ones(len(labels)),  # targets have length 1
            tgt_dict=self.output_vocab,
            left_pad_source=False,
            max_source_positions=self.args.max_input_length,
            max_target_positions=1,
            input_feeding=False,
        )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return (self.args.max_input_length, 1)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.output_vocab






    # We could override this method if we wanted more control over how batches
    # are constructed, but it's not necessary for this tutorial since we can
    # reuse the batching provided by LanguagePairDataset.
    #
    # def get_batch_iterator(
    #     self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
    #     ignore_invalid_inputs=False, required_batch_size_multiple=1,
    #     seed=1, num_shards=1, shard_id=0,
    # ):
    #     (...)


# if __name__ == '__main__':
#     class Args():
#         def __init__(self):
#             self.data = '/home/IAIS/nchakrabor/PycharmProjects/kgqa/datasets/lcquad_dataset/lcquad.en.fairseq'
#             self.xlmr_model_dict = '/home/IAIS/nchakrabor/PycharmProjects/kgqa/kgqa/semparse/experiments/opennmt/embeddings/xlmr/mlm_17_1280.pth'
#             self.max_input_length = 50
#             self.max_output_length = 50
#
#     args = Args()
#     task = SemparseSeq2SeqTask.setup_task(args)
# fairseq-preprocess --trainpref lcquad.en.train --validpref lcquad.en.test --source-lang input --target-lang label --destdir lcquad.en.fairseq --dataset-impl raw
