from pathlib import Path
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.inputters
import onmt.modules
import onmt.utils
import onmt.translate
from onmt.modules import GlobalAttention

from kgqa.onmt.trainer import AdversarialTrainer
from onmt.models import ModelSaver
from onmt.utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Batch

PWD = Path(__file__).parent
PROJECT_ROOT = (PWD / ".." / "..").resolve()  # pylint: disable=no-member
MODULE_ROOT = PROJECT_ROOT / "kgqa"
DATA_ROOT = MODULE_ROOT / "semparse/experiments/opennmt/data"
EMBEDDINGS_ROOT = MODULE_ROOT / "semparse/experiments/opennmt/embeddings"


class LanguageDetector(nn.Module):
    def __init__(self,
                 enc_rnn_size: int,
                 att_context_size: int,
                 num_layers: int,
                 hidden_size: int,
                 dropout: float,
                 batch_norm=False):
        super(LanguageDetector, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'

        # Sentence-level attention network
        self.sentence_attention = nn.Linear(enc_rnn_size, att_context_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(att_context_size, 1,
                                                 bias=False)
        # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector

        self.classifier = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.classifier.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.classifier.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.classifier.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.classifier.add_module('q-relu-{}'.format(i), nn.ReLU())

        self.classifier.add_module('q-linear-final', nn.Linear(hidden_size, 1))

    def forward(self, memory_bank):
        memory_bank = memory_bank.permute(1, 0, 2)
        att_s = self.sentence_attention(memory_bank)
        att_s = torch.tanh(att_s)  # (seq_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s)  # (seq_len)
        align_vectors = F.softmax(att_s, -1).permute(0, 2, 1)
        c = torch.bmm(align_vectors, memory_bank)

        return self.classifier(c)

def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True


def train(bidirectional_encoder: bool,
          enc_rnn_size: int,
          dec_rnn_size: int,
          src_word_vec_size: int,
          tgt_word_vec_size: int,
          dropout: float,
          learning_rate: float,
          train_steps: int,
          valid_steps: int,
          early_stopping_tolerance: int,
          en_preprocessed_data_path: str,
          de_preprocessed_data_path: str,
          save_model_path: str,
          critic_steps: int,
          clip_lower: int,
          clip_upper: int):
    # load en question-query data
    en_vocab_fields = torch.load("{}.vocab.pt".format(en_preprocessed_data_path))

    src_text_field = en_vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = en_vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    # load de unlabelled question data
    de_vocab_fields = torch.load("{}.vocab.pt".format(de_preprocessed_data_path))
    unlabelled_text_field = de_vocab_fields["src"].base_field
    unlabelled_vocab = unlabelled_text_field.vocab
    unlabelled_padding = unlabelled_vocab.stoi[unlabelled_text_field.pad_token]


    # Specify the core model.

    en_encoder_embeddings = onmt.modules.Embeddings(src_word_vec_size, len(src_vocab),
                                                 word_padding_idx=src_padding, dropout=dropout)
    en_encoder_embeddings.load_pretrained_vectors(EMBEDDINGS_ROOT / 'fasttext.en.enc.pt')

    en_encoder = onmt.encoders.RNNEncoder(hidden_size=enc_rnn_size, num_layers=2,
                                       rnn_type="LSTM", bidirectional=bidirectional_encoder, use_bridge=True,
                                       embeddings=en_encoder_embeddings, dropout=dropout)

    de_encoder_embeddings = onmt.modules.Embeddings(src_word_vec_size, len(unlabelled_vocab),
                                                 word_padding_idx=unlabelled_padding, dropout=dropout)
    de_encoder_embeddings.load_pretrained_vectors(EMBEDDINGS_ROOT / 'fasttext.de.enc.pt')

    de_encoder = onmt.encoders.RNNEncoder(hidden_size=enc_rnn_size, num_layers=2,
                                       rnn_type="LSTM", bidirectional=bidirectional_encoder, use_bridge=True,
                                       embeddings=de_encoder_embeddings, dropout=dropout)
    de_encoder.bridge = en_encoder.bridge

    decoder_embeddings = onmt.modules.Embeddings(tgt_word_vec_size, len(tgt_vocab),
                                                 word_padding_idx=tgt_padding, dropout=dropout)
    decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
        hidden_size=dec_rnn_size, num_layers=1, bidirectional_encoder=bidirectional_encoder,
        rnn_type="LSTM", embeddings=decoder_embeddings, dropout=dropout)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = onmt.models.model.NMTModel(en_encoder, decoder)
    de_encoder.to(device)
    model.to(device)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(
        nn.Linear(dec_rnn_size, len(tgt_vocab)),
        nn.LogSoftmax(dim=-1)).to(device)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
        generator=model.generator)

    attn = GlobalAttention(
        enc_rnn_size, coverage=False,
        attn_type='general', attn_func='softmax'
    )
    lang_classifier = LanguageDetector(enc_rnn_size, enc_rnn_size, 2, tgt_word_vec_size, dropout)
    lang_classifier.to(device)

    torch_optimizer = torch.optim.Adam(itertools.chain(de_encoder.parameters(), model.parameters()), lr=learning_rate)
    optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=learning_rate, max_grad_norm=2)
    torch_optimizer_adversary = torch.optim.Adam(itertools.chain(de_encoder.parameters(), lang_classifier.parameters()), lr=learning_rate)

    en_train_data_file = "{}.train.0.pt".format(en_preprocessed_data_path)
    de_train_data_file = "{}.train.0.pt".format(de_preprocessed_data_path)
    en_valid_data_file = "{}.valid.0.pt".format(en_preprocessed_data_path)
    de_valid_data_file = "{}.valid.0.pt".format(de_preprocessed_data_path)

    train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[en_train_data_file],
                                                         fields=en_vocab_fields,
                                                         batch_size=64,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=True,
                                                         repeat=True,
                                                         pool_factor=1)

    en_classifier_train_iter_adversary = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[en_train_data_file],
                                                         fields=en_vocab_fields,
                                                         batch_size=64,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=True,
                                                         repeat=True,
                                                         pool_factor=1)

    de_classifier_train_iter_adversary = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[de_train_data_file],
                                                         fields=de_vocab_fields,
                                                         batch_size=64,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=True,
                                                         repeat=True,
                                                         pool_factor=1)

    de_classifier_train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[de_train_data_file],
                                                                       fields=de_vocab_fields,
                                                                       batch_size=64,
                                                                       batch_size_multiple=1,
                                                                       batch_size_fn=None,
                                                                       device=device,
                                                                       is_train=True,
                                                                       repeat=True,
                                                                       pool_factor=1)

    en_valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[en_valid_data_file],
                                                         fields=en_vocab_fields,
                                                         batch_size=128,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=False,
                                                         repeat=False,
                                                         pool_factor=1)

    de_valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[de_valid_data_file],
                                                         fields=en_vocab_fields,
                                                         batch_size=128,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=False,
                                                         repeat=False,
                                                         pool_factor=1)

    tensorboard = SummaryWriter(flush_secs=5)

    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    report_manager = onmt.utils.ReportMgr(
        report_every=64, tensorboard_writer=tensorboard)
    report_manager.start()

    early_stopper = EarlyStopping(early_stopping_tolerance)
    model_saver = ModelSaver(save_model_path, model, None, en_vocab_fields, optim)
    # adv_model_saver = ModelSaver(save_model_path, model, None, en_vocab_fields, optim)

    en_classifier_train = iter(en_classifier_train_iter_adversary)
    de_classifier_train = iter(de_classifier_train_iter_adversary)

    print("Starting training...")

    def _fix_enc_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        if bidirectional_encoder:
            hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)

        # concat layers and convert to batch x layers*directions*dim
        hidden = hidden.permute(1, 0, 2)

        hidden = hidden.reshape(hidden.size()[0], -1)

        return hidden

    def get_classifier_loss(batch, encoder):
        text, text_lengths = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, None)
        encoder_final, memory_bank, memory_lengths = encoder(text, text_lengths)
        # encoder_final = tuple(_fix_enc_hidden(enc_hid) for enc_hid in encoder_final)
        # features = torch.cat(encoder_final, dim=1)
        classifier_logits = lang_classifier(memory_bank)
        classifier_loss = torch.mean(classifier_logits)
        return classifier_loss

    def train_adversary(trainer):
        freeze_net(model)
        # freeze_net(de_encoder)
        unfreeze_net(lang_classifier)

        n_critic = critic_steps
        task_iter = trainer.optim.training_step
        if critic_steps > 0 and ((task_iter < 50) or (task_iter % 10 == 0)):
            n_critic = 10

        for citer, classifier_en_batch, classifier_de_batch in zip(range(n_critic), en_classifier_train, de_classifier_train):
            for p in lang_classifier.parameters():
                p.data.clamp_(clip_lower, clip_upper)
            lang_classifier.zero_grad()

            classifier_loss = get_classifier_loss(classifier_en_batch, en_encoder)
            (-classifier_loss).backward()

            classifier_loss = get_classifier_loss(classifier_de_batch, de_encoder)
            classifier_loss.backward()

            torch_optimizer_adversary.step()

        unfreeze_net(model)
        # unfreeze_net(de_encoder)
        freeze_net(lang_classifier)

        for p in lang_classifier.parameters():
            p.data.clamp_(clip_lower, clip_upper)

        model.zero_grad()

    def validate_de(trainer):
        print("Validating with de data")
        model.encoder = de_encoder
        stats = trainer.validate(de_valid_iter)
        trainer._report_step(trainer.optim.learning_rate(),
                          trainer.optim.training_step,
                          valid_stats=stats)
        model.encoder = en_encoder

    def eval_en():
        src_reader = onmt.inputters.str2reader["text"]
        tgt_reader = onmt.inputters.str2reader["text"]
        scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7,
                                                 beta=0.,
                                                 length_penalty="avg",
                                                 coverage_penalty="none")
        gpu = 0 if torch.cuda.is_available() else -1
        translator = onmt.translate.Translator(model=model,
                                               fields=en_vocab_fields,
                                               src_reader=src_reader,
                                               tgt_reader=tgt_reader,
                                               global_scorer=scorer,
                                               gpu=gpu)
        builder = onmt.translate.TranslationBuilder(data=torch.load(en_valid_data_file),
                                                    fields=en_vocab_fields,
                                                    has_tgt=True)
        pos_matches = count = 0

        for batch in en_valid_iter:
            trans_batch = translator.translate_batch(
                batch=batch, src_vocabs=[src_vocab],
                attn_debug=False)
            translations = builder.from_batch(trans_batch)
            for trans in translations:
                pred = ' '.join(trans.pred_sents[0])
                gold = ' '.join(trans.gold_sent)
                pos_matches += 1 if pred == gold else 0
                count += 1

        print("Acc: ", pos_matches / count)

    def eval_de():
        src_reader = onmt.inputters.str2reader["text"]
        tgt_reader = onmt.inputters.str2reader["text"]
        scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7,
                                                 beta=0.,
                                                 length_penalty="avg",
                                                 coverage_penalty="none")
        gpu = 0 if torch.cuda.is_available() else -1
        model.encoder = de_encoder
        translator = onmt.translate.Translator(model=model,
                                               fields=de_vocab_fields,
                                               src_reader=src_reader,
                                               tgt_reader=tgt_reader,
                                               global_scorer=scorer,
                                               gpu=gpu)
        builder = onmt.translate.TranslationBuilder(data=torch.load(de_valid_data_file),
                                                    fields=de_vocab_fields,
                                                    has_tgt=True)
        pos_matches = count = 0

        for batch in de_valid_iter:
            trans_batch = translator.translate_batch(
                batch=batch, src_vocabs=[src_vocab],
                attn_debug=False)
            translations = builder.from_batch(trans_batch)
            for trans in translations:
                pred = ' '.join(trans.pred_sents[0])
                gold = ' '.join(trans.gold_sent)
                pos_matches += 1 if pred == gold else 0
                count += 1

        model.encoder = en_encoder

        print("Acc: ", pos_matches / count)


    trainer = AdversarialTrainer(train_adversary=train_adversary,
                            validate_de=validate_de,
                            de_classifier_train_iter=de_classifier_train_iter,
                            get_classifier_loss=get_classifier_loss,
                            en_encoder=en_encoder,
                            de_encoder=de_encoder,
                            lambd=0.5,
                            evalers=[eval_en, eval_de],
                            model=model,
                            train_loss=loss,
                            valid_loss=loss,
                            optim=optim,
                            report_manager=report_manager,
                            dropout=dropout,
                            model_saver=model_saver,
                            earlystopper=early_stopper)

    trainer.train(train_iter=train_iter,
                  train_steps=train_steps,
                  valid_iter=en_valid_iter,
                  valid_steps=valid_steps)



if __name__ == "__main__":
    train(bidirectional_encoder=True,
          enc_rnn_size=512,
          dec_rnn_size=512,
          src_word_vec_size=300,
          tgt_word_vec_size=512,
          dropout=0.6,
          learning_rate=0.001,
          train_steps=4000,
          critic_steps=5,
          clip_lower=-0.01,
          clip_upper=0.01,
          valid_steps=500,
          early_stopping_tolerance=5,
          en_preprocessed_data_path=DATA_ROOT / "simple.productions.out",
          de_preprocessed_data_path=DATA_ROOT / "simple.productions.de.out",
          save_model_path="simple.productions.model")