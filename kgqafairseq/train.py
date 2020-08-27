#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random

import numpy as np
import torch

from fairseq import bleu, checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from torch import autograd
from fairseq_cli import train
from kgqafairseq import SemparseSeq2SeqTask, SemparseClassificationTask, XlmrTransformerEncoderDecoder


class AdvTrainer():
    def __init__(self, model, adv_model):
        self.model = model
        self.adv_model = adv_model


def eval(args, task, models, subset):
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True

    results = []
    hits = 0.
    hit_vector = []

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                out_dict = {}
                if not args.quiet:
                    if src_dict is not None:
                        # print('S-{}\t{}'.format(sample_id, src_str))
                        out_dict['source'] = src_str
                    if has_target:
                        # print('T-{}\t{}'.format(sample_id, target_str))
                        out_dict['gold_target'] = target_str

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        out_dict['pred_target'] = hypo_str
                        # print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        # print('P-{}\t{}'.format(
                        #     sample_id,
                        #     ' '.join(map(
                        #         lambda x: '{:.4f}'.format(x),
                        #         hypo['positional_scores'].tolist(),
                        #     ))
                        # ))

                        # if args.print_alignment:
                            # print('A-{}\t{}'.format(
                            #     sample_id,
                            #     ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                            # ))

                        # if args.print_step:
                            # print('I-{}\t{}'.format(sample_id, hypo['steps']))

                        # if getattr(args, 'retain_iter_history', False):
                            # print("\n".join([
                            #     'E-{}_{}\t{}'.format(
                            #         sample_id, step,
                            #         utils.post_process_prediction(
                            #             h['tokens'].int().cpu(),
                            #             src_str, None, None, tgt_dict, None)[1])
                            #     for step, h in enumerate(hypo['history'])]))

                        # Score only the top hypothesis
                        if has_target and j == 0:
                            if align_dict is not None or args.remove_bpe is not None:
                                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                                target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                            if hasattr(scorer, 'add_string'):
                                scorer.add_string(target_str, hypo_str)
                            else:
                                scorer.add(target_tokens, hypo_tokens)

                    results.append(out_dict)
                    if out_dict['gold_target'] == out_dict['pred_target']:
                        hit_vector.append(1)
                        hits += 1
                    else:
                        hit_vector.append(0)

            wps_meter.update(num_generated_tokens)
            # t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
            num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
        if has_target:
            print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

        acc = hits / len(results)
        print("Hit {}/{}. ACC: {}".format(hits, len(results), acc))

        return acc, hit_vector

# def calc_gradient_penalty(args, critic, source_lang_data, target_lang_data):
#     use_cuda = torch.cuda.is_available() and not args.cpu
#     alpha = torch.rand(args.max_sentences, 1)
#     alpha = alpha.expand(source_lang_data.size())
#     alpha = alpha.cuda() if use_cuda else alpha
#
#     interpolates = alpha * source_lang_data + ((1 - alpha) * target_lang_data)
#
#     if use_cuda:
#         interpolates = interpolates.cuda()
#     interpolates.requires_grad=True
#
#     disc_interpolates = critic(interpolates)
#
#     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
#                                   disc_interpolates.size()),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#     args.gp_gamma = 10
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.gp_gamma
#     return gradient_penalty


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        if "target" not in valid_sub_split:
            task.load_dataset(valid_sub_split, combine=False, epoch=0)
        # task.load_dataset("target_" + valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    assert isinstance(model, XlmrTransformerEncoderDecoder)
    # encoder = model.encoder

    # args.task = 'semparse_classification'
    # adv_task = tasks.setup_task(args, xlmr=task.xlmr)
    # assert isinstance(adv_task, SemparseClassificationTask)
    #
    # # Build adversarial language critic model and criterion (WGAN-GP)
    # adv_model = adv_task.build_model(args)
    # adv_criterion = adv_task.build_criterion(args)
    # print(adv_model)
    # print('| model {}, criterion {}'.format(args.arch, adv_criterion.__class__.__name__))
    # print('| num. model params: {} (num. trained: {})'.format(
    #     sum(p.numel() for p in adv_model.parameters()),
    #     sum(p.numel() for p in adv_model.parameters() if p.requires_grad),
    # ))


    # Build

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    # adv_trainer = Trainer(args, adv_task, adv_model, adv_criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    grad_meter = AverageMeter()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')

    if args.plot_features:
        valid_losses, accuracy = validate(args, trainer, task, epoch_itr, valid_subsets)
        plot_features(args, trainer, task, epoch_itr, valid_subsets, accuracy=accuracy)

    if args.validate_first:
        validate(args, trainer, task, epoch_itr, valid_subsets)

    while (
        lr > args.min_lr
        and (epoch_itr.epoch < max_epoch or (epoch_itr.epoch == max_epoch
            and epoch_itr._next_epoch_itr is not None))
        and trainer.get_num_updates() < max_update
    ):
        # train for one epoch
        train(args, trainer, task, epoch_itr, grad_meter)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses, accuracy = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        reload_dataset = ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(epoch_itr.epoch, load_dataset=reload_dataset)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr, grad_meter):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        exclude_print = {'wps', 'ups', 'wpb', 'bsz', 'lr', 'clip', 'oom', 'wall'}
        progress._log_to_tensorboard(stats, 'train', stats['num_updates'])
        progress.wrapped_bar.print({key:value for (key,value) in stats.items() if key not in exclude_print}, tag='train', step=stats['num_updates'])
        # progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second and updates-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()
            trainer.get_meter('ups').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses, accuracy = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats

def get_features(args, task, models, subset):
    use_cuda = torch.cuda.is_available() and not args.cpu

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    features = []

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            tokens = sample['net_input']['src_tokens']
            if tokens.size()[1] == 10:
                padding = torch.LongTensor([]).new_full((tokens.size()[0], 1), fill_value=0).cuda()
                tokens = torch.cat([tokens, padding], dim=1)
                sample['net_input']['src_tokens'] = tokens

            encoder = models[0].encoder
            with torch.no_grad():
                features.append(encoder(**sample['net_input']).encoder_out)

    return features


def plot_features(args, trainer, task, epoch_itr, subsets, accuracy):
    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    features = {}
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=None,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        subset_features = get_features(args, task, models=[trainer.get_model()], subset=subset)
        last_features = subset_features[-1]
        size = last_features.size()
        padding_samples = 64 - size[1]
        subset_features[-1] = torch.cat([last_features, torch.zeros((size[0], padding_samples, size[2])).cuda()], dim=1)
        features[subset] = subset_features
        accuracy[subset] += padding_samples*[-1]

    source = features['valid']
    target = features['target-valid']
    accuracy = accuracy['valid'] + accuracy['target-valid']
    everything = source + target
    features = torch.nn.utils.rnn.pad_sequence(everything)
    features = features.view(features.size()[0], -1, features.size()[3])
    total_samples = features.size()[1]
    features = features.permute(1, 0, 2).reshape(total_samples, -1)

    import pickle
    features_out = args.feature_file
    with open(features_out, 'wb') as fp:
        pickle.dump({'features': features.cpu().numpy(), 'accuracy': accuracy}, fp)


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    accuracy = dict()

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=None,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        acc, hits = eval(args, task, models=[trainer.get_model()], subset=subset)
        accuracy[subset] = hits
        extra_meters[subset + "_acc"].update(acc)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )

    return valid_losses, accuracy


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)

def get_training_and_generation_parser(default_task='translation'):
    parser = options.get_parser('Trainer', default_task)
    options.add_dataset_args(parser, train=True, gen=True)
    options.add_generation_args(parser)
    options.add_distributed_training_args(parser)
    options.add_model_args(parser)
    options.add_optimization_args(parser)
    options.add_checkpoint_args(parser)
    return parser

def cli_main():
    parser = get_training_and_generation_parser()
    args = options.parse_args_and_arch(parser)
    args.validate_first = False
    args.plot_features = False
    args.feature_file = "features.wganlp.out"

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
