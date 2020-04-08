from pathlib import Path
import fastBPE
from fairseq.data import Dictionary

PWD = Path(__file__).parent
PROJECT_ROOT = (PWD / ".." / "..").resolve()  # pylint: disable=no-member
MODULE_ROOT = PROJECT_ROOT / "kgqa"
DATA_ROOT = MODULE_ROOT / "semparse/experiments/opennmt/data"
EMBEDDINGS_ROOT = MODULE_ROOT / "semparse/experiments/opennmt/embeddings"
XLMR_ROOT = EMBEDDINGS_ROOT / "xlmr"

class BPEEncoder():
    def __init__(self, code_file, vocab_file, dictionary: Dictionary):
        self.bpe = fastBPE.fastBPE(str(code_file), str(vocab_file))
        self.dictionary = dictionary
        self.n_w = 0
        self.n_oov = 0

    def reset_counts(self):
        self.n_w = 0
        self.n_oov = 0

    def to_bpe(self, sentence):
        sentence_bpe = self.bpe.apply([sentence])[0]
        self.n_w += len(sentence_bpe.split())
        self.n_oov += len([w for w in sentence_bpe.split() if w not in self.dictionary.indices])

        # add </s> sentence delimiters
        sentence_bpe = '</s> {} </s>'.format(sentence_bpe.strip())

        return sentence_bpe