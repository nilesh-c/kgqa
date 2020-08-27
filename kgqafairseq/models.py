from collections import OrderedDict

import torch
from fairseq.data import Dictionary
from torch import nn
from torch.nn import functional as F
from fairseq.models import FairseqEncoderDecoderModel, register_model, FairseqDecoder, FairseqEncoder, BaseFairseqModel
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder, Embedding, EncoderOut
from fairseq.models.lstm import LSTMDecoder, LSTMEncoder
from fairseq.models import register_model_architecture
from fairseq.models.roberta import RobertaHubInterface
import fairseq.utils
import numpy as np
from kgqafairseq.language_transfer import LanguageTransferTask
from kgqafairseq.tasks import SemparseClassificationTask
from fairseq_cli import generate


from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.models.transformer import Linear
import random

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


class TransformerEncoder2(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])


    def forward(self, encoder_out, cls_input=None, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        encoder_embedding = None

        # B x T x C -> T x B x C
        x = encoder_out


        encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                print('deleting {0}'.format(weights_key))
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict






















def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False

def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True

class TransformerCritic(FairseqDecoder):
    def __init__(self,
                 enc_out_size: int,
                 att_context_size: int,
                 num_layers: int,
                 input_size: int,
                 hidden_size: int,
                 dropout: float,
                 dictionary: Dictionary,
                 args,
                 batch_norm=False,
                 output_scalar=True):
        super(TransformerCritic, self).__init__(dictionary)
        self.transformer_encoder = TransformerEncoder2(args, dictionary, None)

        assert num_layers >= 0, 'Invalid layer numbers'

        # Sentence-level attention network
        self.sentence_attention = nn.Linear(enc_out_size, att_context_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(att_context_size, 1,
                                                 bias=False)
        # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector

        self.classifier = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.classifier.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.classifier.add_module('q-linear-{}'.format(i),
                                       nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            if batch_norm:
                self.classifier.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.classifier.add_module('q-relu-{}'.format(i), nn.ReLU())

        output_size = 1 if output_scalar else len(self.dictionary)
        self.classifier.add_module('q-linear-final', nn.Linear(hidden_size, output_size))

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """

        @param memory_bank: EncoderOut
        @return: classifier logits
        """
        features = self.transformer_encoder(encoder_out.encoder_out, features_only=True).encoder_out
        att_s = self.sentence_attention(features)
        att_s = torch.tanh(att_s)  # (seq_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s)  # (seq_len)
        align_vectors = F.softmax(att_s, -1).permute(0, 2, 1)
        c = torch.bmm(align_vectors, features)
        # self.classifier.to(c.device)
        logits = self.classifier(c.squeeze())  # .view(-1)
        return (logits, None)


class EncoderOutClassifier(FairseqDecoder):
    def __init__(self,
                 args,
                 enc_out_size: int,
                 att_context_size: int,
                 num_layers: int,
                 input_size: int,
                 hidden_size: int,
                 dropout: float,
                 dictionary: Dictionary,
                 batch_norm=False,
                 output_scalar=True):
        super(EncoderOutClassifier, self).__init__(dictionary)
        assert num_layers >= 0, 'Invalid layer numbers'

        self.logitcond = args.logitcond
        self.conditional = args.conditional
        if self.conditional:
            # input_size = 512
            # self.random_projection_size = 70
            # self.output_sequence_size = 60
            # input_size = self.random_projection_size*self.output_sequence_size
            input_size = input_size + 768

        # args.encoder_embed_dim = 176
        # self.transformer_encoder = TransformerEncoder2(args, dictionary, None)
        # input_size += 176

        # Sentence-level attention network
        self.sentence_attention = nn.Linear(enc_out_size, att_context_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(att_context_size, 1,
                                                 bias=False)

        # Sentence-level attention network
        self.task_output_attention = nn.Linear(768, att_context_size)

        # Sentence context vector to take dot-product with
        self.task_output_context_vector = nn.Linear(att_context_size, 1,
                                                 bias=False)
        # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector

        # self.random_projection_size = 512
        # self.random_f = torch.randn((enc_out_size, self.random_projection_size)).cuda()
        # self.random_g = torch.randn((60*176, self.random_projection_size)).cuda()

        # self.random_f = torch.randn((1, enc_out_size, self.random_projection_size)).cuda()
        # self.random_f = self.random_f.repeat(self.output_sequence_size, 1, 1)
        # self.random_g = torch.randn((176, self.random_projection_size)).cuda()
        # self.random_g = self.random_g.repeat(self.output_sequence_size, 1, 1)

        # self.decoder_linear = nn.Linear(60*176, 512)
        # self.decoder_relu = nn.ReLU()

        # self.f_proj = nn.Sequential()
        # if dropout > 0:
        #     self.f_proj.add_module('fproj-dropout', nn.Dropout(p=dropout))
        # self.f_proj.add_module('fproj-linear', nn.Linear(enc_out_size, 256))
        #
        # self.g_proj = nn.Sequential()
        # if dropout > 0:
        #     self.g_proj.add_module('fproj-dropout', nn.Dropout(p=dropout))
        # self.g_proj.add_module('fproj-linear', nn.Linear(176, 256))

        self.classifier = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.classifier.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.classifier.add_module('q-linear-{}'.format(i), nn.Linear(input_size if i==0 else hidden_size, hidden_size))
            if batch_norm:
                self.classifier.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.classifier.add_module('q-relu-{}'.format(i), nn.ReLU())

        output_size = 1 if output_scalar else len(self.dictionary)
        self.classifier.add_module('q-linear-final', nn.Linear(hidden_size, output_size))

    def encode_main_task_output(self, decoder_out_features):
        # features = self.transformer_encoder(main_task_output, features_only=True).encoder_out
        features = decoder_out_features
        att_s = self.task_output_attention(features)
        att_s = torch.tanh(att_s)  # (seq_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.task_output_context_vector(att_s)  # (seq_len)
        align_vectors = F.softmax(att_s, -1).permute(0, 2, 1)
        c = torch.bmm(align_vectors, features)
        return c.squeeze()

        # features = self.pad_sequences(decoder_out_features).reshape(-1, 60 * 176)
        # features = self.decoder_relu(self.decoder_linear(features))
        # return features

    def interaction_function(
            self,
            f: torch.FloatTensor,
            g: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Evaluate the interaction function for given embeddings.
        The embeddings have to be in a broadcastable shape.
        :param h: shape: (batch_size, num_entities, d)
            Head embeddings.
        :param r: shape: (batch_size, num_entities, d)
            Relation embeddings.
        :param t: shape: (batch_size, num_entities, d)
            Tail embeddings.
        :return: shape: (batch_size, num_entities)
            The scores.
        """
        # Circular correlation of entity embeddings
        a_fft = torch.rfft(f, signal_ndim=1, onesided=True)
        b_fft = torch.rfft(g, signal_ndim=1, onesided=True)

        # complex conjugate, a_fft.shape = (batch_size, num_entities, d', 2)
        a_fft[:, :, 1] *= -1

        # Hadamard product in frequency domain
        p_fft = a_fft * b_fft

        # inverse real FFT, shape: (batch_size, num_entities, d)
        composite = torch.irfft(p_fft, signal_ndim=1, onesided=True, signal_sizes=(f.shape[-1],))

        return composite

        return scores

    def pad_sequences(self, tensor, max_seq_len=60, padding_value=0.):
        new_size = [i for i in tensor.size()]
        # assuming batch first
        new_size[1] = max_seq_len - new_size[1]
        if new_size[1] == 0:
            return tensor

        padding = tensor.new_full(new_size, padding_value).cuda()
        tensor = torch.cat([tensor, padding], dim=1)
        return tensor

    def forward(self, prev_output_tokens, encoder_out=None, decoder_out_features=None, **kwargs):
        if self.conditional:
            if self.logitcond:
                return self.logit_cond(prev_output_tokens, encoder_out=encoder_out, decoder_out_features=decoder_out_features, **kwargs)
            return self.multi_random_cond(prev_output_tokens, encoder_out=encoder_out, decoder_out_features=decoder_out_features, **kwargs)
        else:
            return self.nocond(prev_output_tokens, encoder_out=encoder_out, decoder_out_features=decoder_out_features, **kwargs)

    def multi_random_cond(self, prev_output_tokens, encoder_out=None, decoder_out_features=None, **kwargs):
        """

                @param memory_bank: EncoderOut
                @return: classifier logits
                """
        memory_bank = encoder_out.encoder_out
        memory_bank = memory_bank.permute(1, 0, 2)
        att_s = self.sentence_attention(memory_bank)
        att_s = torch.tanh(att_s)  # (seq_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s)  # (seq_len)
        align_vectors = F.softmax(att_s, -1).permute(0, 2, 1)
        c = torch.bmm(align_vectors, memory_bank).squeeze()
        # self.classifier.to(c.device)
        encoded_proj = torch.matmul(c.unsqueeze(1).unsqueeze(1), self.random_f).squeeze()
        decoder_out_features = self.pad_sequences(decoder_out_features)
        decoded_proj = torch.matmul(decoder_out_features.unsqueeze(2), self.random_g).squeeze()

        # decoder_out_features = self.main_decoder(prev_output_tokens, encoder_out=encoder_out)[0]
        features = torch.mul(encoded_proj, decoded_proj) / np.sqrt(self.random_projection_size)
        features = features.view(-1, self.random_projection_size*self.output_sequence_size)
        # features = self.interaction_function(encoded_proj, decoded_proj) / np.sqrt(self.random_projection_size)

        # features = torch.cat([c.squeeze(), self.encode_main_task_output(decoder_out_features)], dim=1) if self.conditional else c.squeeze()
        logits = self.classifier(features)
        return (logits, None)

    def random_cond(self, prev_output_tokens, encoder_out=None, decoder_out_features=None, **kwargs):
        """

        @param memory_bank: EncoderOut
        @return: classifier logits
        """
        memory_bank = encoder_out.encoder_out
        memory_bank = memory_bank.permute(1, 0, 2)
        att_s = self.sentence_attention(memory_bank)
        att_s = torch.tanh(att_s)  # (seq_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s)  # (seq_len)
        align_vectors = F.softmax(att_s, -1).permute(0, 2, 1)
        c = torch.bmm(align_vectors, memory_bank).squeeze()
        # self.classifier.to(c.device)
        encoded_proj = torch.matmul(c, self.random_f)
        decoded_proj = torch.matmul(self.pad_sequences(decoder_out_features).reshape(-1, 60*176), self.random_g)
        # decoder_out_features = self.main_decoder(prev_output_tokens, encoder_out=encoder_out)[0]
        features = torch.mul(encoded_proj, decoded_proj) / np.sqrt(self.random_projection_size)
        # features = self.interaction_function(encoded_proj, decoded_proj) / np.sqrt(self.random_projection_size)

        # features = torch.cat([c.squeeze(), self.encode_main_task_output(decoder_out_features)], dim=1) if self.conditional else c.squeeze()
        logits = self.classifier(features)
        return (logits, None)

    def logit_cond(self, prev_output_tokens, encoder_out=None, decoder_out_features=None, **kwargs):
        """

        @param memory_bank: EncoderOut
        @return: classifier logits
        """
        memory_bank = encoder_out.encoder_out
        memory_bank = memory_bank.permute(1, 0, 2)
        att_s = self.sentence_attention(memory_bank)
        att_s = torch.tanh(att_s)  # (seq_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s)  # (seq_len)
        align_vectors = F.softmax(att_s, -1).permute(0, 2, 1)
        c = torch.bmm(align_vectors, memory_bank).squeeze()
        features = torch.cat([c.squeeze(), self.encode_main_task_output(decoder_out_features)], dim=1) if self.conditional else c.squeeze()
        logits = self.classifier(features)
        return (logits, None)

    def nocond(self, prev_output_tokens, encoder_out=None, decoder_out_features=None, **kwargs):
        """

        @param memory_bank: EncoderOut
        @return: classifier logits
        """
        memory_bank = encoder_out.encoder_out
        memory_bank = memory_bank.permute(1, 0, 2)
        att_s = self.sentence_attention(memory_bank)
        att_s = torch.tanh(att_s)  # (seq_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s)  # (seq_len)
        align_vectors = F.softmax(att_s, -1).permute(0, 2, 1)
        c = torch.bmm(align_vectors, memory_bank)
        # self.classifier.to(c.device)

        # decoder_out_features = self.main_decoder(prev_output_tokens, encoder_out=encoder_out)[0]

        features = torch.cat([c.squeeze(), self.encode_main_task_output(decoder_out_features)], dim=1) if self.conditional else c.squeeze()

        # f_features = self.f_proj(c.squeeze())
        # g_features = self.g_proj(self.encode_main_task_output(decoder_out_features))

        # features = self.interaction_function(f_features, g_features)
        # features = torch.cat([f_features, g_features], dim=1) if self.conditional else c.squeeze()

        logits = self.classifier(features)
        return (logits, None)

class XlmrEncoder(FairseqEncoder):
    def __init__(self, xlmr: RobertaHubInterface, max_pos):
        dictionary = xlmr.task.dictionary
        super().__init__(dictionary)
        self.xlmr = xlmr
        self.model = xlmr.model
        self.max_pos = max_pos
        self.padding_idx = dictionary.pad()

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        if src_tokens.dim() == 1:
            tokens = src_tokens.unsqueeze(0)
        if src_tokens.size(-1) > self.model.max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                src_tokens.size(-1), self.model.max_positions()
            ))
        features, inner_states = self.model(
            src_tokens,
            features_only=True,
            return_all_hiddens=True,
        )
        inner_states = inner_states['inner_states']

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return EncoderOut(
            encoder_out=features,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=inner_states,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out


@register_model("language_classifier")
class LanguageClassificationModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add a new command-line argument to configure the
        # dimensionality of the hidden state.
        parser.add_argument(
            '--classifier-att-context-size', type=int, metavar='N',
            help='number of dimensions of attention context in classification model',
        )

        parser.add_argument(
            '--classifier-hidden-size', type=int, metavar='N',
            help='number of dimensions of hidden layers in classification model',
        )

        parser.add_argument(
            '--classifier-num-layers', type=int, metavar='N',
            help='number of hidden layers in classification model',
        )

    @classmethod
    def build_model(cls, args, task: LanguageTransferTask):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a FairseqRNNClassifier instance.

        # Initialize XLM-R encoder
        xlmr = XlmrEncoder(task.xlmr, task.max_positions()[0])

        # Initialize our classifier module
        classifier = EncoderOutClassifier(
            args=args,
            enc_out_size=768,
            att_context_size=args.classifier_att_context_size,
            num_layers=args.classifier_num_layers,
            input_size=args.classifier_input_size,
            hidden_size=args.classifier_hidden_size,
            dropout=args.dropout,
            dictionary=task.output_vocab
        )

        # Return the wrapped version of the module
        model = LanguageClassificationModel(xlmr, classifier)
        model.main_decoder = task.main_model.decoder
        model.conditional = args.conditional
        model.logitcond = args.logitcond

        # Print the model architecture.
        print(model)

        return model

    def train_main(self):
        self.training_main = True
        unfreeze_net(self.encoder.xlmr)
        unfreeze_net(self.encoder.xlmr.model)
        freeze_net(self.decoder)

    def train_adversary(self):
        self.training_main = False
        freeze_net(self.encoder.xlmr)
        freeze_net(self.encoder.xlmr.model)
        unfreeze_net(self.decoder)


    def forward(self, src_tokens, src_lengths, prev_output_tokens=None, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        if not self.training_main:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.detach()
            )
        def get_normalized_probs(prev_output_tokens, encoder_out):
            logits = self.main_decoder(prev_output_tokens, encoder_out=encoder_out)
            if not self.logitcond:
                softmax = self.main_decoder.get_normalized_probs(logits, sample=None, log_probs=False).detach()
                return softmax
            else:
                # return logits[0]
                return logits[1]['inner_states'][-1].permute(1,0,2)

        decoder_out_features = get_normalized_probs(prev_output_tokens, encoder_out) if self.conditional else None
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, decoder_out_features=decoder_out_features, **kwargs)
        return decoder_out

    # def __init__(self, encoder: XlmrEncoder, classifier: EncoderOutClassifier):
    #     super(LanguageClassificationModel, self).__init__()
    #
    #     self.encoder = encoder
    #     self.classifier = classifier
    #
    # def forward(self, src_tokens, src_lengths):
    #     # self.encoder.to(src_tokens.device)
    #     # self.classifier.to(src_tokens.device)
    #     encoder_out = self.encoder(src_tokens, src_lengths)
    #     logits = self.classifier(encoder_out)
    #     return logits


# class XlmrEncoder(FairseqDecoder):
#     def __init__(self, xlmr: RobertaHubInterface, max_pos):
#         dictionary = xlmr.task.dictionary
#         super().__init__(dictionary)
#         # self.model = xlmr.model
#         self.max_pos = max_pos
#         self.padding_idx = dictionary.pad()
#         self.decoder = xlmr.model.decoder
#         assert isinstance(self.decoder, FairseqDecoder)
#
#     def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
#         self.decoder.forward(prev_output_tokens)
#
#     def reorder_encoder_out(self, encoder_out, new_order):
#         """
#         Reorder encoder output according to *new_order*.
#
#         Args:
#             encoder_out: output from the ``forward()`` method
#             new_order (LongTensor): desired order
#
#         Returns:
#             *encoder_out* rearranged according to *new_order*
#         """
#         if encoder_out.encoder_out is not None:
#             encoder_out = encoder_out._replace(
#                 encoder_out=encoder_out.encoder_out.index_select(1, new_order)
#             )
#         if encoder_out.encoder_padding_mask is not None:
#             encoder_out = encoder_out._replace(
#                 encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
#             )
#         if encoder_out.encoder_embedding is not None:
#             encoder_out = encoder_out._replace(
#                 encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
#             )
#         if encoder_out.encoder_states is not None:
#             for idx, state in enumerate(encoder_out.encoder_states):
#                 encoder_out.encoder_states[idx] = state.index_select(1, new_order)
#         return encoder_out
#
#     def max_positions(self):
#         return self.max_pos


@register_model('xlmr_transformer_model')
class XlmrTransformerEncoderDecoder(FairseqEncoderDecoderModel):

    encoder: XlmrEncoder
    decoder: FairseqDecoder

    # def max_positions(self):
    #     """Maximum length supported by the model."""
    #     return OrderedDict([('labeled', (self.encoder.max_positions(), self.decoder.max_positions())),
    #                         ('unlabeled', (self.encoder.max_positions(), self.decoder.max_positions()))])

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--encoder-dropout', type=float, default=0.4,
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
            '--decoder-dropout', type=float, default=0.4,
            help='decoder dropout probability',
        )

    @classmethod
    def build_model(cls, args, task: SemparseClassificationTask):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.
        xlmr = XlmrEncoder(task.xlmr, task.max_positions()[0])
        # encoder = LSTMEncoder(
        #     dictionary=task.source_dictionary,
        #     pretrained_embed=xlmr,
        #     embed_dim=args.xlmr_out_dim,
        #     hidden_size=args.decoder_hidden_dim,
        #     dropout_in=args.decoder_dropout,
        #     dropout_out=args.decoder_dropout
        # )

        from fairseq.models.transformer_from_pretrained_xlm import TransformerDecoderFromPretrainedXLM
        dictionary = task.output_vocab
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        embed_tokens = Embedding(num_embeddings, args.decoder_embed_dim, padding_idx)
        decoder = TransformerDecoder(args, dictionary, embed_tokens)

        # decoder = LSTMDecoder(
        #     dictionary=task.target_dictionary,
        #     encoder_output_units=args.encoder_hidden_dim,
        #     embed_dim=args.decoder_embed_dim,
        #     hidden_size=args.decoder_hidden_dim,
        #     dropout_in=args.decoder_dropout,
        #     dropout_out=args.decoder_dropout
        # )
        model = XlmrTransformerEncoderDecoder(xlmr, decoder)

        # Print the model architecture.
        print(model)

        return model

    def freeze_encoder(self):
        freeze_net(self.encoder.xlmr)
        freeze_net(self.encoder.xlmr.model)

    def unfreeze_encoder(self):
        unfreeze_net(self.encoder.xlmr)
        unfreeze_net(self.encoder.xlmr.model)

    def train_main(self):
        self.training_main = True
        unfreeze_net(self.encoder.xlmr)
        unfreeze_net(self.encoder.xlmr.model)
        unfreeze_net(self.decoder)

    def train_adversary(self):
        self.training_main = False
        freeze_net(self.encoder.xlmr)
        freeze_net(self.encoder.xlmr.model)
        freeze_net(self.decoder)

    # We could override the ``forward()`` if we wanted more control over how
    # the encoder and decoder interact, but it's not necessary for this
    # tutorial since we can inherit the default implementation provided by
    # the FairseqEncoderDecoderModel base class, which looks like:
    #
        # def forward(self, src_tokens, src_lengths, prev_output_tokens):
        #     encoder_out = self.encoder(src_tokens, src_lengths)
        #     decoder_out = self.decoder(prev_output_tokens, encoder_out)
        #     return decoder_out


@register_model_architecture('xlmr_transformer_model', 'xlmr_test')
def simple_seq2seq2(args):
    # args.xlmr_out_dim = getattr(args, 'xlmr_out_dim', 512)
    args.xlmr_out_dim = getattr(args, 'xlmr_out_dim', 768)
    # args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.xlmr_out_dim)
    # args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 512)

    # Transformer arch
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.xlmr_out_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.decoder_layerdrop = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.xlmr_out_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.xlmr_out_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)
    args.max_target_positions = getattr(args, 'max_target_positions', args.max_output_length)


    args.max_epoch = 40
    args.no_epoch_checkpoints = True
    # args.no_save = True
    args.save_interval = 10
    args.patience = 5
    args.lr_scheduler = 'inverse_sqrt'
    args.warmup_init_lr = 0.00001
    args.warmup_updates = 500
    args.optimizer = 'adam'
    # args.fp16 = True
    args.lr = [0.0001]
    args.lr = [0.0001]
    args.learning_rate = [0.0001]

@register_model_architecture('xlmr_transformer_model', 'langtransfer_test')
def simple_seq2seq(args):
    # # Language classifier stuff
    # args.classifier_att_context_size = getattr(args, 'classifier_att_context_size', 512)
    # args.classifier_num_layers = getattr(args, 'classifier_num_layers', 2)
    # args.classifier_hidden_size = getattr(args, 'classifier_hidden_size', 768)

    # args.xlmr_out_dim = getattr(args, 'xlmr_out_dim', 512)
    args.xlmr_out_dim = getattr(args, 'xlmr_out_dim', 768)
    args.conditional = getattr(args, 'conditional', False)
    args.logitcond = getattr(args, 'conditional', True)
    # args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.xlmr_out_dim)
    # args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 512)

    # Transformer arch
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.1)

    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.xlmr_out_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.2)
    args.decoder_layerdrop = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.xlmr_out_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.xlmr_out_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)
    args.max_target_positions = getattr(args, 'max_target_positions', args.max_output_length)

    args.wasserstein_coeff = 0.1
    args.gp_gamma = 10.

    args.max_epoch = 200
    args.no_epoch_checkpoints = True
    args.no_save = True
    args.save_interval = 10
    args.patience = 5
    #
    # args.lr_scheduler = 'cosine'
    # args.warmup_init_lr = 0.000001
    # args.lr_period_updates = 200
    # args.max_lr = 0.00005
    # args.warmup_updates = 200

    args.lr_scheduler = 'inverse_sqrt'
    args.warmup_init_lr = 0.00001
    args.warmup_updates = 500
    args.optimizer = 'adam'
    #args.fp16 = False
    args.lr = [0.0001]
    args.learning_rate = [0.0001]

@register_model_architecture('language_classifier', 'lang_test')
def language_classification(args):
    args.classifier_att_context_size = getattr(args, 'classifier_att_context_size', 128)
    args.classifier_num_layers = getattr(args, 'classifier_num_layers', 1)
    args.classifier_input_size = getattr(args, 'classifier_input_size', 768)
    args.classifier_hidden_size = getattr(args, 'classifier_hidden_size', 128)
    args.dropout = getattr(args, 'dropout', 0.1)

    args.max_epoch = 40
    args.no_epoch_checkpoints = True
    # args.no_save = True
    # args.save_interval = 5
    args.patience = 5
    args.lr_scheduler = 'inverse_sqrt'
    args.warmup_init_lr = 0.00001
    args.warmup_updates = 500
    args.optimizer = 'adam'
    #args.fp16 = False
    args.lr = [0.0001]
    args.learning_rate = [0.0001]
