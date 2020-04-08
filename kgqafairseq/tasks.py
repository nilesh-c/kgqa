import os
import torch

from fairseq.data import Dictionary, LanguagePairDataset
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
        parser.add_argument('--xlmr-model-dict', metavar='XLMR_PATH',
                            help='file path for torch-serialized xlmr model and dictionary')
        parser.add_argument('--max-input-length', default=50, type=int,
                            help='max input utterance length')
        parser.add_argument('--max-output-length', default=50, type=int,
                            help='max output logical form length')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just load the Dictionaries.
        reloaded = torch.load(args.xlmr_model_dict)

        # build dictionary / update parameters
        # input_vocab = XlmrDictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
        input_vocab = Dictionary()
        input_vocab.indices = reloaded['dico_word2id']
        input_vocab.symbols = list(input_vocab.indices.keys())
        input_vocab.count = [reloaded['dico_counts'][word] for word in input_vocab.symbols]
        # for word, count in reloaded['dico_counts'].items():
        #     input_vocab.indices[word] = len(input_vocab.symbols)
        #     input_vocab.symbols.append(word)
        #     input_vocab.count.append(count)

        label_vocab = Dictionary.load(os.path.join(args.data, 'dict.label.txt'))
        print('| [input] dictionary: {} types'.format(len(input_vocab)))
        print('| [label] dictionary: {} types'.format(len(label_vocab)))

        params = AttrDict(reloaded['params'])
        params.n_words = len(input_vocab.symbols)
        params.bos_index = input_vocab.index(BOS_WORD)
        params.eos_index = input_vocab.index(EOS_WORD)
        params.pad_index = input_vocab.index(PAD_WORD)
        params.unk_index = input_vocab.index(UNK_WORD)
        params.mask_index = input_vocab.index(MASK_WORD)

        return SemparseSeq2SeqTask(args, input_vocab, label_vocab, params, reloaded['model'])

    def __init__(self, args, input_vocab, output_vocab, dict_params, encoder_state_dict):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.dict_params = dict_params
        self.encoder_state_dict = encoder_state_dict

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        prefix = os.path.join(self.args.data, '{}.input-label'.format(split))

        # Read input sentences.
        input_utterances, input_lengths = [], []
        with open(prefix + '.input', encoding='utf-8') as file:
            for line in file:
                input_utterance = line.strip()

                # Tokenize the sentence, splitting on spaces
                tokens = self.input_vocab.encode_line(
                    input_utterance, add_if_not_exist=False,
                )

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
                )

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


if __name__ == '__main__':
    class Args():
        def __init__(self):
            self.data = '/home/IAIS/nchakrabor/PycharmProjects/kgqa/datasets/lcquad_dataset/lcquad.en.fairseq'
            self.xlmr_model_dict = '/home/IAIS/nchakrabor/PycharmProjects/kgqa/kgqa/semparse/experiments/opennmt/embeddings/xlmr/mlm_17_1280.pth'
            self.max_input_length = 50

    args = Args()
    task = SemparseSeq2SeqTask.setup_task(args)
