# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


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
    """

    def __init__(self,
                 batch_size,
                 epoch_queue_starts=15,
                 crops_for_assign=[0, 1],
                 feat_dim=128,
                 queue_length=0,
                 interval=1,
                 **kwargs):
        self.batch_size = batch_size * dist.get_world_size()\
            if dist.is_initialized() else batch_size
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.feat_dim = feat_dim
        self.queue_length = queue_length
        self.interval = interval
        self.queue = None

    def before_run(self, runner):
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

    def before_train_epoch(self, runner):
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

    def after_train_epoch(self, runner):
        self.queue = runner.model.module.head.queue

        if self.queue is not None and self.every_n_epochs(
                runner, self.interval):
            torch.save({'queue': self.queue}, self.queue_path)
