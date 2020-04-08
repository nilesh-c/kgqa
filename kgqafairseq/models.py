from fairseq.models import FairseqEncoderDecoderModel, register_model
from fairseq.models.lstm import LSTMDecoder
from fairseq.models import register_model_architecture
from kgqafairseq.tasks import SemparseSeq2SeqTask
from kgqafairseq.src.model.transformer import TransformerModel


@register_model('xlmr_lstm_model')
class XlmrLstmModel(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--encoder-dropout', type=float, default=0.1,
            help='encoder dropout probability',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )

    @classmethod
    def build_model(cls, args, task: SemparseSeq2SeqTask):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.
        encoder = TransformerModel(task.dict_params, True, True)
        encoder.eval()
        encoder.load_state_dict(task.encoder_state_dict)

        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            encoder_output_units=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_dim,
            dropout_in=args.decoder_dropout,
            dropout_out=args.decoder_dropout
        )
        model = XlmrLstmModel(encoder, decoder)

        # Print the model architecture.
        print(model)

        return model

    # We could override the ``forward()`` if we wanted more control over how
    # the encoder and decoder interact, but it's not necessary for this
    # tutorial since we can inherit the default implementation provided by
    # the FairseqEncoderDecoderModel base class, which looks like:
    #
    # def forward(self, src_tokens, src_lengths, prev_output_tokens):
    #     encoder_out = self.encoder(src_tokens, src_lengths)
    #     decoder_out = self.decoder(prev_output_tokens, encoder_out)
    #     return decoder_out


@register_model_architecture('xlmr_lstm_model', 'xlmr_test')
def simple_seq2seq(args):
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 1280)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
