import torch
import traceback

import onmt.utils
from onmt import Trainer
from onmt.utils.logging import logger

class AdversarialTrainer(Trainer):


    def __init__(self, train_adversary,
              validate_de,
              de_classifier_train_iter,
              get_classifier_loss,
              en_encoder,
              de_encoder,
              lambd,
              evalers, model, train_loss, valid_loss, optim, trunc_size=0, shard_size=32, norm_method="sents",
                 accum_count=[1], accum_steps=[0], n_gpu=1, gpu_rank=1, gpu_verbose_level=0, report_manager=None,
                 with_align=False, model_saver=None, average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0]):
        super().__init__(model, train_loss, valid_loss, optim, trunc_size, shard_size, norm_method, accum_count,
                         accum_steps, n_gpu, gpu_rank, gpu_verbose_level, report_manager, with_align, model_saver,
                         average_decay, average_every, model_dtype, earlystopper, dropout, dropout_steps)
        self.train_adversary = train_adversary
        self.validate_de = validate_de
        self.de_classifier_train_iter = de_classifier_train_iter
        self.get_classifier_loss = get_classifier_loss
        self.en_encoder = en_encoder
        self.de_encoder = de_encoder
        self.lambd = lambd
        self.evalers = evalers

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        for i, ((batches, normalization), (de_batches, de_normalization)) in enumerate(
                zip(self._accum_batches(train_iter), self._accum_batches(self.de_classifier_train_iter))):
            self.train_adversary(self)

            step = self.optim.training_step
            # UPDATE DROPOUT
            self._maybe_update_dropout(step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                normalization = sum(onmt.utils.distributed
                                    .all_gather_list
                                    (normalization))

            self._gradient_accumulation(
                batches, de_batches, normalization, total_stats,
                report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._update_average(step)

            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate(
                    valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                    step %d' % (self.gpu_rank, step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

                self.validate_de(self)

                for func in self.evalers:
                    func()

            if (self.model_saver is not None
                    and (save_checkpoint_steps != 0
                         and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step, moving_average=self.moving_average)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step, moving_average=self.moving_average)
        return total_stats


    def _gradient_accumulation(self, true_batches, true_batches_de, normalization, total_stats,
                               report_stats):
        if self.accum_count > 1:
            self.optim.zero_grad()

        for k, (batch, de_batch) in enumerate(zip(true_batches, true_batches_de)):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt


            # get batches for computing adversary loss
            en_batch = batch

            bptt = False
            for j in range(0, target_size - 1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.optim.zero_grad()

                outputs, attns = self.model(src, tgt, src_lengths, bptt=bptt,
                                            with_align=self.with_align)
                bptt = True

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size)

                    if loss is not None:
                        # self.optim.backward(loss)
                        loss.backward(retain_graph=True)

                    # adversary losses
                    classifier_loss_en = self.get_classifier_loss(en_batch, self.en_encoder)
                    classifier_loss_de = self.get_classifier_loss(de_batch, self.de_encoder)
                    (self.lambd * (classifier_loss_en - classifier_loss_de)).backward()

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d",
                                self.optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.model.parameters()
                                 if p.requires_grad
                                 and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(
                            grads, float(1))
                    self.optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.model.decoder.state is not None:
                    self.model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()