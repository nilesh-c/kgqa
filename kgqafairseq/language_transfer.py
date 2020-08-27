import math
import os
from collections import OrderedDict

import torch
from fairseq import optim, utils
from fairseq.data import Dictionary, LanguagePairDataset, MonolingualDataset, RoundRobinZipDatasets, TokenBlockDataset, \
    EpochBatchIterator, iterators, data_utils, FairseqDataset

from fairseq.models.roberta import RobertaHubInterface
from fairseq.models.transformer import EncoderOut
from fairseq.tasks import FairseqTask, register_task
from matplotlib.lines import Line2D
from torch import autograd
import numpy as np
from matplotlib import pyplot as plt
import json
from torch.utils.data import TensorDataset


def plot_grad_flow(fp, model_name, named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if (p.requires_grad) and hasattr(p.grad, 'abs') and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    graddict = {}
    for i in ('layers', 'ave_grads', 'max_grads'):
        graddict[i] = locals()[i]

    fp.write(json.dumps((model_name, graddict)) + "\n")


@register_task('language_transfer')
class LanguageTransferTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='SOURCE_DATA_PATH',
                            help='file prefix for data')
        parser.add_argument('--target-data',  default='', type=str,
                            help='file prefix for low-resource target language data')
        parser.add_argument('--target2source-data', default='', type=str,
                            help='file prefix for machine-translated-to-source version of low-resource target language data')
        parser.add_argument('--max-input-length', default=50, type=int,
                            help='max input utterance length')
        parser.add_argument('--max-output-length', default=50, type=int,
                            help='max output logical form length')
        parser.add_argument('--gp', default=False, type=bool,
                            help='gradient penalty if set, Lipschitz penalty otherwise')
        parser.add_argument('--conditional', default=False, type=bool,
                            help='use conditional dan if set')
        parser.add_argument('--baseline', default=False, type=bool,
                            help='Train baseline main model without adversarial loop')
        parser.add_argument('--gp-gamma', default=10, type=float,
                            help='gradient penalty coefficient')
        parser.add_argument('--wasserstein-coeff', default=0.1, type=float,
                            help='wasserstein loss coefficient for main task')

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

        return LanguageTransferTask(args, input_vocab, output_vocab, xlmr)

    def __init__(self, args, input_vocab, output_vocab, xlmr: RobertaHubInterface):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.xlmr = xlmr
        self.burnedin = False
        self.baseline = args.baseline
        self.conditional = args.conditional
        self.logitcond = args.logitcond
        self.steps = 0
        print("Training with {} regularization".format("GP" if args.gp else "LP"))

    # def load_dataset(self, split, **kwargs):
    #     source_semparse = self.load_source_semparse_dataset(split)
    #     target = self.load_target_dataset(split)
    #
    #     self.datasets[split] = RoundRobinZipDatasets(OrderedDict([('labeled', source_semparse), ('unlabeled', target)]))

    def load_dataset(self, split, **kwargs):
        self.datasets[split] = self.load_source_dataset(split)
        if self.args.target_data != '':
            self.datasets['target-' + split] = self.load_target_dataset(self.args.target_data, split)
        if self.args.target2source_data  != '':
            self.datasets['target2source-' + split] = self.load_target_dataset(self.args.target2source_data, split)

    def load_source_dataset(self, split, **kwargs):
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
        dataset = LanguagePairDataset(
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
        return dataset


    def load_target_dataset(self, target_data, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        prefix = os.path.join(target_data, '{}.input-label'.format(split))

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

        print('| {} {} {} examples'.format(self.args.target_data, split, len(input_utterances)))

        # We reuse LanguagePairDataset since classification can be modeled as a
        # sequence-to-sequence task where the target sequence has length 1.
        dataset = LanguagePairDataset(
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
        return dataset

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_input_length, self.args.max_output_length)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.output_vocab

    def get_local_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        return epoch_iter

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        if 'target-train' not in self.datasets or not self.datasets['target-train']:
            # self.load_target_dataset('train')
            if self.args.target_data != '':
                target_dataset = self.load_target_dataset('train')
        target_dataset = self.dataset('target-train')
        self.target_batch_iter_generator: EpochBatchIterator = super().get_batch_iterator(target_dataset, max_tokens, max_sentences, max_positions,
                                                                                          ignore_invalid_inputs, required_batch_size_multiple, seed,
                                                                                          num_shards, shard_id, num_workers, epoch)
        self.target_batch_iter = self.target_batch_iter_generator.next_epoch_itr()
        main_dataset_local = self.load_source_dataset('train')
        self.main_batch_iter_generator: EpochBatchIterator = self.get_local_batch_iterator(main_dataset_local   , max_tokens, max_sentences, max_positions, ignore_invalid_inputs, required_batch_size_multiple, seed, num_shards, shard_id, num_workers, epoch)

        return super().get_batch_iterator(dataset, max_tokens, max_sentences, max_positions, ignore_invalid_inputs, required_batch_size_multiple, seed, num_shards, shard_id, num_workers, epoch)

    def build_model(self, args):
        self.main_model = super().build_model(args)

        if not self.baseline:
            from copy import deepcopy
            model_args = deepcopy(args)
            model_args.classifier_att_context_size = getattr(args, 'classifier_att_context_size', 512)
            model_args.classifier_num_layers = getattr(args, 'classifier_num_layers', 2)
            model_args.classifier_input_size = getattr(args, 'classifier_input_size', 768)
            model_args.classifier_hidden_size = getattr(args, 'classifier_hidden_size', 512)
            model_args.fp16 = True
            model_args.arch = 'lang_test'
            self.adversary_model = super().build_model(model_args)
            # self.adversary_model.half()
            # self.adversary_model.decoder.half()

            use_cuda = torch.cuda.is_available() and not self.args.cpu
            if use_cuda:
                self.adversary_model.cuda()
            from copy import deepcopy
            optim_args = deepcopy(self.args)
            optim_args.lr = [0.0001]
            optim_args.learning_rate = [0.0001]
            self._adversarial_optimizer = optim.build_optimizer(optim_args, self.adversary_model.decoder.parameters())

            optim_args = deepcopy(self.args)
            optim_args.lr = [0.00001]
            optim_args.learning_rate = [0.00001]
            self._wasserstein_optimizer = optim.build_optimizer(optim_args, self.main_model.encoder.parameters())

        return self.main_model

    def train_main_model(self, source_sample, target_sample, model, criterion, optimizer, ignore_grad=False):
        model.train()

        # wasserstein_loss = self.wasserstein_loss(source_sample, target_sample)
        # print("Wasserstein loss: {}".format(wasserstein_loss))
        # (self.args.wasserstein_coeff * wasserstein_loss).backward()
        # self._wasserstein_optimizer.step()

        wasserstein_loss = 0 if self.baseline else self.wasserstein_loss(source_sample, target_sample)

        loss, sample_size, logging_output = criterion(model, source_sample)

        # logging_output['gen_w_loss'] = wasserstein_loss.item()

        if ignore_grad:
            loss *= 0

        if self.baseline:
            loss.backward()
        else:
            (self.args.wasserstein_coeff * wasserstein_loss + loss).backward()


        return loss, sample_size, logging_output

    def create_conditional_features(self, sample):
        with torch.no_grad():
            main_task_output = self.main_model(**sample['net_input'])[0].detach()
        return {**sample['net_input'], **{"main_task_output": main_task_output}}

    def wasserstein_loss(self, source_sample, target_sample):
        source_sample['target'], target_sample['target'] = self.pad_equal(source_sample['target'], target_sample['target'])
        source_sample['net_input']['prev_output_tokens'], target_sample['net_input']['prev_output_tokens'] = \
            self.pad_equal(source_sample['net_input']['prev_output_tokens'],
                           target_sample['net_input']['prev_output_tokens'])

        source_sample['net_input']['src_tokens'], target_sample['net_input']['src_tokens'] = \
            self.pad_equal(source_sample['net_input']['src_tokens'],
                           target_sample['net_input']['src_tokens'])

        source_scores = self.adversary_model(**source_sample['net_input'])[0].mean()
        target_scores = self.adversary_model(**target_sample['net_input'])[0].mean()
        return (source_scores - target_scores)

    def train_adversary(self, source_sample, target_sample):
        # if not self.args.gp:
        #     for p in self.adversary_model.decoder.parameters():
        #         p.data.clamp_(-0.01, 0.01)
        self._adversarial_optimizer.zero_grad()
        self.adversary_model.decoder.zero_grad()
        wasserstein_loss = self.wasserstein_loss(source_sample, target_sample)

        grad_penalty = self.calc_gradient_penalty(source_sample, target_sample)
        final_cost = grad_penalty - wasserstein_loss
        self._adversarial_optimizer.backward(final_cost)
        # plot_grad_flow("gp-wasserstein cost", final_cost.named_parameters())
        self._adversarial_optimizer.step()

        self._adversarial_optimizer.zero_grad()
        self.adversary_model.decoder.zero_grad()
        return wasserstein_loss, grad_penalty, final_cost

    def pad_equal(self, source_tokens, target_tokens):
        use_cuda = torch.cuda.is_available() and not self.args.cpu

        padding_idx = self.input_vocab.pad()
        source_size = source_tokens.size()
        target_size = target_tokens.size()
        extra_seq_len = abs(target_size[1] - source_size[1])
        extra_batch_size = abs(target_size[0] - source_size[0])

        if source_size[0] < target_size[0]:
            target_tokens = target_tokens[extra_batch_size:]
        elif source_size[0] > target_size[0]:
            source_tokens = source_tokens[extra_batch_size:]

        batch_size = source_tokens.size()[0]

        def get_padding():
            padding = torch.LongTensor([]).new_full((batch_size, extra_seq_len), fill_value=padding_idx)
            return padding.cuda() if use_cuda else padding

        if source_size[1] < target_size[1]:
            source_tokens = torch.cat((source_tokens, get_padding()), dim=1)
        elif source_size[1] > target_size[1]:
            target_tokens = torch.cat((target_tokens, get_padding()), dim=1)

        return source_tokens, target_tokens

    def calc_gradient_penalty(self, source_sample, target_sample):
        use_cuda = torch.cuda.is_available() and not self.args.cpu

        source_tokens = source_sample['net_input']['src_tokens']
        target_tokens = target_sample['net_input']['src_tokens']

        source_tokens, target_tokens = self.pad_equal(source_tokens, target_tokens)
        batch_size = source_tokens.size()[0]

        # source_sample_net_input = self.create_conditional_features(source_sample)
        # target_sample_net_input = self.create_conditional_features(target_sample)

        source_lang_features = self.adversary_model.encoder(src_tokens=source_tokens, src_lengths=source_sample['net_input']['src_lengths'])
        target_lang_features = self.adversary_model.encoder(src_tokens=target_tokens, src_lengths=target_sample['net_input']['src_lengths'])

        alpha_encoded = torch.rand(batch_size, 1)
        alpha_encoded = alpha_encoded.expand(source_lang_features.encoder_out.size())
        alpha_encoded = alpha_encoded.cuda() if use_cuda else alpha_encoded

        encoded_interpolates = alpha_encoded * source_lang_features.encoder_out + ((1 - alpha_encoded) * target_lang_features.encoder_out)

        if use_cuda:
            encoded_interpolates = encoded_interpolates.cuda()
        encoded_interpolates.requires_grad=True

        if self.conditional:
            def get_normalized_probs(prev_output_tokens, encoder_out):
                logits = self.main_model.decoder(prev_output_tokens, encoder_out=encoder_out)
                if not self.logitcond:
                    softmax = self.main_model.decoder.get_normalized_probs(logits, sample=None, log_probs=False).permute(1, 0, 2).detach()
                    return softmax
                else:
                    # return logits[0].permute(1, 0, 2)
                    return logits[1]['inner_states'][-1]

            source_lang_decoder_out = get_normalized_probs(source_sample['net_input']['prev_output_tokens'],
                                                              encoder_out=source_lang_features)
            target_lang_decoder_out = get_normalized_probs(target_sample['net_input']['prev_output_tokens'],
                                                           encoder_out=target_lang_features)

            alpha_decoded = torch.rand(batch_size, 1)
            alpha_decoded = alpha_decoded.expand(source_lang_decoder_out.size())
            alpha_decoded = alpha_decoded.cuda() if use_cuda else alpha_decoded
            decoded_interpolates = alpha_decoded * source_lang_decoder_out + ((1 - alpha_decoded) * target_lang_decoder_out)
            decoded_interpolates = decoded_interpolates.permute(1, 0, 2)

            if use_cuda:
                decoded_interpolates = decoded_interpolates.cuda()
            decoded_interpolates.requires_grad = True
        else:
            decoded_interpolates = None

        encoder_out = EncoderOut(
            encoder_out=encoded_interpolates,  # T x B x C
            encoder_padding_mask=None,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=None,  # List[T x B x C]
        )

        # self.adversary_model.decoder.half()
        critic_interpolates = self.adversary_model.decoder(source_sample['net_input']['prev_output_tokens'], encoder_out=encoder_out, decoder_out_features=decoded_interpolates)[0]

        gradients = autograd.grad(outputs=critic_interpolates, inputs=[encoded_interpolates, decoded_interpolates] if self.logitcond else encoded_interpolates,
                                  grad_outputs=torch.ones(critic_interpolates.size()).cuda() if use_cuda else torch.ones(
                                      critic_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        import torch.nn.functional as F
        if self.args.gp:
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.args.gp_gamma
        else:
            gradient_penalty = ((F.relu(gradients.norm(2, dim=1) - 1)) ** 2).mean() * self.args.gp_gamma
        return gradient_penalty


    # def calc_gradient_penalty(self, source_sample, target_sample):
    #     use_cuda = torch.cuda.is_available() and not self.args.cpu
    #
    #     source_tokens = source_sample['net_input']['src_tokens']
    #     target_tokens = target_sample['net_input']['src_tokens']
    #
    #     source_tokens, target_tokens = self.pad_equal(source_tokens, target_tokens)
    #     batch_size = source_tokens.size()[0]
    #
    #     # source_sample_net_input = self.create_conditional_features(source_sample)
    #     # target_sample_net_input = self.create_conditional_features(target_sample)
    #
    #     source_lang_features = self.adversary_model.encoder(src_tokens=source_tokens, src_lengths=source_sample['net_input']['src_lengths']).encoder_out
    #     target_lang_features = self.adversary_model.encoder(src_tokens=target_tokens, src_lengths=target_sample['net_input']['src_lengths']).encoder_out
    #
    #     alpha_encoded = torch.rand(batch_size, 1).half()
    #     # alpha_decoded = alpha_encoded.clone()
    #
    #     alpha_encoded = alpha_encoded.expand(source_lang_features.size())
    #     alpha_encoded = alpha_encoded.cuda() if use_cuda else alpha_encoded
    #
    #     interpolates = alpha_encoded * source_lang_features + ((1 - alpha_encoded) * target_lang_features)
    #
    #     # source_lang_output = source_sample_net_input['main_task_output'].permute(1, 0, 2)
    #     # target_lang_output = target_sample_net_input['main_task_output'].permute(1, 0, 2)
    #     # alpha_decoded = alpha_decoded.expand(source_lang_output.size())
    #     # alpha_decoded = alpha_decoded.cuda() if use_cuda else alpha_decoded
    #     # main_task_output_interpolates = alpha_decoded * source_lang_output + ((1 - alpha_decoded) * target_lang_output)
    #
    #     if use_cuda:
    #         interpolates = interpolates.cuda()
    #     interpolates.requires_grad=True
    #
    #     encoder_out = EncoderOut(
    #         encoder_out=interpolates,  # T x B x C
    #         encoder_padding_mask=None,  # B x T
    #         encoder_embedding=None,  # B x T x C
    #         encoder_states=None,  # List[T x B x C]
    #     )
    #
    #     # self.adversary_model.decoder.half()
    #     critic_interpolates = self.adversary_model.decoder(source_sample['net_input']['prev_output_tokens'], encoder_out=encoder_out)[0]
    #
    #     gradients = autograd.grad(outputs=critic_interpolates, inputs=interpolates,
    #                               grad_outputs=torch.ones(critic_interpolates.size()).cuda() if use_cuda else torch.ones(
    #                                   critic_interpolates.size()),
    #                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    #
    #     import torch.nn.functional as F
    #     if self.args.gp:
    #         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.args.gp_gamma
    #     else:
    #         gradient_penalty = ((F.relu(gradients.norm(2, dim=1) - 1)) ** 2).mean() * self.args.gp_gamma
    #     return gradient_penalty

    # def adversarial_optimizer(self):
    #     if not self._adversarial_optimizer:
    #         # from copy import deepcopy
    #         # optim_args = deepcopy(self.args)
    #         # optim_args.lr = [0.0001]
    #         # optim_args.learning_rate = [0.0001]
    #         self._adversarial_optimizer = optim.build_optimizer(self.args, self.adversary_model.parameters())
    #
    #     return self._adversarial_optimizer

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        source_sample = sample
        self.steps += 1
        target_sample = None

        if not self.baseline:
            if not self.burnedin:
                self.adversary_model.train_adversary()
                self.main_model.train_adversary()
                self.main_batch_iter = self.main_batch_iter_generator.next_epoch_itr()
                for i in range(20):
                    print("Burning in critic, epoch {}".format(i))
                    while self.main_batch_iter.has_next() and self.target_batch_iter.has_next():
                        source_sample = next(self.main_batch_iter)
                        target_sample = next(self.target_batch_iter)
                        use_cuda = torch.cuda.is_available() and not self.args.cpu
                        source_sample = utils.move_to_cuda(source_sample) if use_cuda else source_sample
                        target_sample = utils.move_to_cuda(target_sample) if use_cuda else target_sample
                        wasserstein_loss, grad_penalty, final_cost = self.train_adversary(source_sample, target_sample)
                        # print("Wasserstein loss: {}, grad penalty: {}, final loss: {}".format(wasserstein_loss, grad_penalty.item(), final_cost.item()))

                    self.target_batch_iter = self.target_batch_iter_generator.next_epoch_itr()
                    self.main_batch_iter = self.main_batch_iter_generator.next_epoch_itr()

                self.burnedin = True


            #self.adversarial_optimizer()

            # if self.steps % len(self.main_batch_iter) == 0:
            self.adversary_model.train_adversary()
            self.main_model.train_adversary()
            self.main_batch_iter = self.main_batch_iter_generator.next_epoch_itr()
            wgan_stats = {'critic_w_sum': 0., 'critic_gp_sum': 0., 'critic_final_sum': 0.}

            if not self.target_batch_iter.has_next():
                self.target_batch_iter = self.target_batch_iter_generator.next_epoch_itr()
            target_sample = next(self.target_batch_iter)
            use_cuda = torch.cuda.is_available() and not self.args.cpu
            target_sample = utils.move_to_cuda(target_sample) if use_cuda else target_sample

            for i in range(10):
                # if not self.main_batch_iter.has_next():
                #     self.main_batch_iter = self.main_batch_iter_generator.next_epoch_itr()
                # if not self.target_batch_iter.has_next():
                #     self.target_batch_iter = self.target_batch_iter_generator.next_epoch_itr()
                # source_sample = next(self.main_batch_iter)
                # target_sample = next(self.target_batch_iter)
                wasserstein_loss, grad_penalty, final_cost = self.train_adversary(source_sample, target_sample)
                wgan_stats['critic_w_sum'] += wasserstein_loss.item()
                wgan_stats['critic_gp_sum'] += grad_penalty.item()
                wgan_stats['critic_final_sum'] += final_cost.item()
                # print("Wasserstein loss: {}, grad penalty: {}, final loss: {}".format(wasserstein_loss,
                #                                                                       grad_penalty.item(),
                #                                                                       final_cost.item()))


            self.adversary_model.train_main()

        self.main_model.train_main()
        loss, sample_size, logging_output = self.train_main_model(source_sample, target_sample, model, criterion, optimizer, ignore_grad)

        if not self.baseline:
            for k, v in wgan_stats.items():
                logging_output[k] = v

        return loss, sample_size, logging_output

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        custom_stats = ['critic_w_sum', 'critic_gp_sum', 'critic_final_sum', 'gen_w_loss', 'BLAHTEST']
        getsum = lambda x: sum(log.get(x, 0) for log in logging_outputs)
        custom_stats = {k: getsum(k) for k in custom_stats}

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size
        }

        agg_output.update(custom_stats)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
