from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('cosine_restarts')
class CosineScheduleWithRestarts(FairseqLRScheduler):
    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with cosine.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        warmup_end_lr = args.max_lr
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = args.lr[0]

        self.min_lr = args.lr[0]
        self.max_lr = args.max_lr

        assert self.max_lr > self.min_lr, 'max_lr must be more than lr'

        self.t_mult = args.t_mult
        self.period = args.lr_period_updates

        if self.period <= 0:
            assert args.max_update >= 0, 'Either --max_update or --lr-period-updates must be set'
            self.period = args.max_update - args.warmup_updates

        if args.warmup_updates > 0:
            # linearly warmup for the first args.warmup_updates
            self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates
        else:
            self.lr_step = 1

        self.warmup_updates = args.warmup_updates
        self.lr_shrink = args.lr_shrink

        # initial learning rate
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--max-lr', type=float, metavar='LR',
                            help='max learning rate, must be more than args.lr')
        parser.add_argument('--t-mult', default=1, type=float, metavar='LR',
                            help='factor to grow the length of each period')
        parser.add_argument('--lr-period-updates', default=-1, type=float, metavar='LR',
                            help='initial number of updates per period')
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing')