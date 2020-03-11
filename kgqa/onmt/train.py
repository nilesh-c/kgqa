from pathlib import Path

import torch
import torch.nn as nn

import onmt
import onmt.inputters
import onmt.modules
import onmt.utils
import onmt.translate
from onmt.models import ModelSaver
from onmt.utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

PWD = Path(__file__).parent
PROJECT_ROOT = (PWD / ".." / "..").resolve()  # pylint: disable=no-member
MODULE_ROOT = PROJECT_ROOT / "kgqa"
DATA_ROOT = MODULE_ROOT / "semparse/experiments/opennmt/data"

def train(enc_rnn_size: int,
          dec_rnn_size: int,
          src_word_vec_size: int,
          tgt_word_vec_size: int,
          dropout: float,
          learning_rate: float,
          train_steps: int,
          valid_steps: int,
          early_stopping_tolerance: int,
          preprocessed_data_path: str,
          save_model_path: str):
    vocab_fields = torch.load("{}.vocab.pt".format(preprocessed_data_path))

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    # Specify the core model.

    encoder_embeddings = onmt.modules.Embeddings(src_word_vec_size, len(src_vocab),
                                                 word_padding_idx=src_padding, dropout=dropout)

    encoder = onmt.encoders.RNNEncoder(hidden_size=enc_rnn_size, num_layers=2,
                                       rnn_type="LSTM", bidirectional=True,
                                       embeddings=encoder_embeddings, dropout=dropout)

    decoder_embeddings = onmt.modules.Embeddings(tgt_word_vec_size, len(tgt_vocab),
                                                 word_padding_idx=tgt_padding, dropout=dropout)
    decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
        hidden_size=dec_rnn_size, num_layers=1, bidirectional_encoder=True,
        rnn_type="LSTM", embeddings=decoder_embeddings, dropout=dropout,
        attn_type=None)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = onmt.models.model.NMTModel(encoder, decoder)
    model.to(device)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(
        nn.Linear(dec_rnn_size, len(tgt_vocab)),
        nn.LogSoftmax(dim=-1)).to(device)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
        generator=model.generator)

    torch_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=learning_rate, max_grad_norm=2)

    # Load some data
    from itertools import chain
    train_data_file = "{}.train.0.pt".format(preprocessed_data_path)
    valid_data_file = "{}.valid.0.pt".format(preprocessed_data_path)
    train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[train_data_file],
                                                         fields=vocab_fields,
                                                         batch_size=64,
                                                         batch_size_multiple=1,
                                                         batch_size_fn=None,
                                                         device=device,
                                                         is_train=True,
                                                         repeat=True,
                                                         pool_factor=1)

    valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[valid_data_file],
                                                         fields=vocab_fields,
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
    model_saver = ModelSaver(save_model_path, model, None, vocab_fields, optim)

    trainer = onmt.Trainer(model=model,
                           train_loss=loss,
                           valid_loss=loss,
                           optim=optim,
                           report_manager=report_manager,
                           dropout=dropout,
                           model_saver=model_saver,
                           earlystopper=early_stopper)

    print("Starting training...")
    trainer.train(train_iter=train_iter,
                  train_steps=train_steps,
                  valid_iter=valid_iter,
                  valid_steps=valid_steps)

    src_reader = onmt.inputters.str2reader["text"]
    tgt_reader = onmt.inputters.str2reader["text"]
    scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7,
                                             beta=0.,
                                             length_penalty="avg",
                                             coverage_penalty="none")
    gpu = 0 if torch.cuda.is_available() else -1
    translator = onmt.translate.Translator(model=model,
                                           fields=vocab_fields,
                                           src_reader=src_reader,
                                           tgt_reader=tgt_reader,
                                           global_scorer=scorer,
                                           gpu=gpu)
    builder = onmt.translate.TranslationBuilder(data=torch.load(valid_data_file),
                                                fields=vocab_fields,
                                                has_tgt=True)
    pos_matches = count = 0

    for batch in valid_iter:
        trans_batch = translator.translate_batch(
            batch=batch, src_vocabs=[src_vocab],
            attn_debug=False)
        translations = builder.from_batch(trans_batch)
        for trans in translations:
            pred = ' '.join(trans.pred_sents[0])
            gold = ' '.join(trans.gold_sent)
            pos_matches += 1 if pred==gold else 0
            count += 1

    print("Acc: ", pos_matches/count)

if __name__ == "__main__":
    train(enc_rnn_size=512,
          dec_rnn_size=512,
          src_word_vec_size=512,
          tgt_word_vec_size=512,
          dropout=0.6,
          learning_rate=0.001,
          train_steps=4000,
          valid_steps=500,
          early_stopping_tolerance=5,
          preprocessed_data_path=DATA_ROOT / "simple.productions.out",
          save_model_path="simple.productions.model")