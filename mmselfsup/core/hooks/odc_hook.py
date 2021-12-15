# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.runner import HOOKS, Hook
from mmcv.utils import print_log


@HOOKS.register_module()
class ODCHook(Hook):
    """Hook for ODC.

    This hook includes the online clustering process in ODC.

    Args:
        centroids_update_interval (int): Frequency of iterations
            to update centroids.
        deal_with_small_clusters_interval (int): Frequency of iterations
            to deal with small clusters.
        evaluate_interval (int): Frequency of iterations to evaluate clusters.
        reweight (bool): Whether to perform loss re-weighting.
        reweight_pow (float): The power of re-weighting.
        dist_mode (bool): Use distributed training or not. Defaults to True.
    """

    def __init__(self,
                 centroids_update_interval,
                 deal_with_small_clusters_interval,
                 evaluate_interval,
                 reweight,
                 reweight_pow,
                 dist_mode=True):
        assert dist_mode, 'non-dist mode is not implemented'
        self.centroids_update_interval = centroids_update_interval
        self.deal_with_small_clusters_interval = \
            deal_with_small_clusters_interval
        self.evaluate_interval = evaluate_interval
        self.reweight = reweight
        self.reweight_pow = reweight_pow

    def after_train_iter(self, runner):
        # centroids update
        if self.every_n_iters(runner, self.centroids_update_interval):
            runner.model.module.memory_bank.update_centroids_memory()

        # deal with small clusters
        if self.every_n_iters(runner, self.deal_with_small_clusters_interval):
            runner.model.module.memory_bank.deal_with_small_clusters()

        # reweight
        runner.model.module.set_reweight()

        # evaluate
        if self.every_n_iters(runner, self.evaluate_interval):
            new_labels = runner.model.module.memory_bank.label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            self.evaluate(runner, new_labels.numpy())

    def after_train_epoch(self, runner):
        # save cluster
        if self.every_n_epochs(runner, 10) and runner.rank == 0:
            new_labels = runner.model.module.memory_bank.label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            np.save(f'{runner.work_dir}/cluster_epoch_{runner.epoch + 1}.npy',
                    new_labels.numpy())

    def evaluate(self, runner, new_labels):
        histogram = np.bincount(
            new_labels, minlength=runner.model.module.memory_bank.num_classes)
        empty_cls = (histogram == 0).sum()
        minimal_cls_size, maximal_cls_size = histogram.min(), histogram.max()
        if runner.rank == 0:
            print_log(
                f'empty_num: {empty_cls.item()}\t'
                f'min_cluster: {minimal_cls_size.item()}\t'
                f'max_cluster:{maximal_cls_size.item()}',
                logger='root')
