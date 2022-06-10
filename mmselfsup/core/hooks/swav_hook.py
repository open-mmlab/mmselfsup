# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from mmengine.hooks import Hook

from mmselfsup.registry import HOOKS


@HOOKS.register_module()
class SwAVHook(Hook):
    """Hook for SwAV.

    This hook builds the queue in SwAV according to ``epoch_queue_starts``.
    The queue will be saved in ``runner.work_dir`` or loaded at start epoch
    if the path folder has queues saved before.

    Args:
        batch_size (int): the batch size per GPU for computing.
        epoch_queue_starts (int, optional): from this epoch, starts to use the
            queue. Defaults to 15.
        crops_for_assign (list[int], optional): list of crops id used for
            computing assignments. Defaults to [0, 1].
        feat_dim (int, optional): feature dimension of output vector.
            Defaults to 128.
        queue_length (int, optional): length of the queue (0 for no queue).
            Defaults to 0.
        interval (int, optional): the interval to save the queue.
            Defaults to 1.
        frozen_layers_cfg (dict, optional): Dict to config frozen layers.
            The key-value pair is layer name and its frozen iters. If frozen,
            the layers don't need gradient. Defaults to dict().
    """

    def __init__(
        self,
        batch_size: int,
        epoch_queue_starts: Optional[int] = 15,
        crops_for_assign: Optional[List[int]] = [0, 1],
        feat_dim: Optional[int] = 128,
        queue_length: Optional[int] = 0,
        interval: Optional[int] = 1,
        frozen_layers_cfg: Optional[Dict] = dict()
    ) -> None:
        self.batch_size = batch_size * dist.get_world_size()\
            if dist.is_initialized() else batch_size
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.feat_dim = feat_dim
        self.queue_length = queue_length
        self.interval = interval
        self.frozen_layers_cfg = frozen_layers_cfg
        self.requires_grad = True
        self.queue = None

    def before_run(self, runner) -> None:
        if dist.is_initialized():
            self.queue_path = osp.join(runner.work_dir,
                                       'queue' + str(dist.get_rank()) + '.pth')
        else:
            self.queue_path = osp.join(runner.work_dir, 'queue.pth')
        # build the queue
        if osp.isfile(self.queue_path):
            self.queue = torch.load(self.queue_path)['queue']
            runner.model.module.head.queue = self.queue
        # the queue needs to be divisible by the batch size
        self.queue_length -= self.queue_length % self.batch_size

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        for layer, frozen_iters in self.frozen_layers_cfg.items():
            if runner.iter < frozen_iters and self.requires_grad:
                self.requires_grad = False
                for name, p in runner.model.module.named_parameters():
                    if layer in name:
                        p.requires_grad = False
            elif runner.iter >= frozen_iters and not self.requires_grad:
                self.requires_grad = True
                for name, p in runner.model.module.named_parameters():
                    if layer in name:
                        p.requires_grad = True

    def before_train_epoch(self, runner) -> None:
        # optionally starts a queue
        if self.queue_length > 0 \
            and runner.epoch >= self.epoch_queue_starts \
                and self.queue is None:
            self.queue = torch.zeros(
                len(self.crops_for_assign),
                self.queue_length // runner.world_size,
                self.feat_dim,
            ).cuda()

        # set the boolean type of use_the_queue
        runner.model.module.head.queue = self.queue
        runner.model.module.head.use_queue = False

    def after_train_epoch(self, runner) -> None:
        self.queue = runner.model.module.head.queue

        if self.queue is not None and self.every_n_epochs(
                runner, self.interval):
            torch.save({'queue': self.queue}, self.queue_path)
