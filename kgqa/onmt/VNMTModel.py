""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn
from onmt.models import NMTModel


class VNMTModel(NMTModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, lstm_hidden_size, latent_size):
        super(VNMTModel, self).__init__(encoder, decoder)

        self.latent_size = latent_size

        self.context_to_mu = nn.Linear(
            lstm_hidden_size,
            latent_size)
        self.context_to_logvar = nn.Linear(
            lstm_hidden_size,
            latent_size)
        self.lstm_state2context = nn.Linear(
            2 * lstm_hidden_size,
            latent_size)

    def get_hidden(self, state):
        hidden, context = state[0][-1], state[1][-1]
        hidden = self.lstm_state2context(torch.cat([hidden, context], -1))
        return hidden

    def reparameterize(self, encoder_state):
        """
        context [B x 2H]
        """
        hidden = self.get_hidden(encoder_state)
        mu = self.context_to_mu(hidden)
        logvar = self.context_to_logvar(hidden)
        if self.training:
            std = torch.mul(logvar, 0.5).exp()
            eps = torch.normal(0., 1., size=std.size())
            z = eps.mul(std).add(mu)
        else:
            z = mu
        return z, mu, logvar

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)

        # mean pooling encoder hidden states
        pooled_state = torch.mean()

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(dec_in, memory_bank,
                                      memory_lengths=lengths,
                                      with_align=with_align)
        return dec_out, attns

        # encoding side
        encoder_outputs, encoder_state = self.encoder(
            self.src_embedding(src), src_lengths)

        # re-parameterize
        z, mu, logvar = self.reparameterize(encoder_state)
        # encoder to decoder
        decoder_state = self.encoder2decoder(encoder_state)

        trg_feed = trg[:-1]
        decoder_input = torch.cat([
            self.trg_embedding(trg_feed),
            z.unsqueeze(0).repeat(trg_feed.size(0), 1, 1)],
            -1)

        # decoding side
        decoder_outputs, decoder_state, attns = self.decoder(
            decoder_input, encoder_outputs, src_lengths, decoder_state)

        return decoder_outputs, decoder_state, attns, compute_kld(mu, logvar)

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
