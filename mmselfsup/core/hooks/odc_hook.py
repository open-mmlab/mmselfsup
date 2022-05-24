# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmselfsup.registry import HOOKS


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
                 centroids_update_interval: int,
                 deal_with_small_clusters_interval: int,
                 evaluate_interval: int,
                 reweight: bool,
                 reweight_pow: float,
                 dist_mode: Optional[bool] = True) -> None:
        assert dist_mode, 'non-dist mode is not implemented'
        self.centroids_update_interval = centroids_update_interval
        self.deal_with_small_clusters_interval = \
            deal_with_small_clusters_interval
        self.evaluate_interval = evaluate_interval
        self.reweight = reweight
        self.reweight_pow = reweight_pow

    def after_train_iter(self, runner) -> None:
        # centroids update
        if self.every_n_iters(runner, self.centroids_update_interval):
            runner.model.module.memory_bank.update_centroids_memory()

        # deal with small clusters
        if self.every_n_iters(runner, self.deal_with_small_clusters_interval):
            runner.model.module.memory_bank.deal_with_small_clusters()

        # reweight
        self.set_reweight(runner)

        # evaluate
        if self.every_n_iters(runner, self.evaluate_interval):
            new_labels = runner.model.module.memory_bank.label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            self.evaluate(runner, new_labels.numpy())

    def after_train_epoch(self, runner) -> None:
        # save cluster
        if self.every_n_epochs(runner, 10) and runner.rank == 0:
            new_labels = runner.model.module.memory_bank.label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            np.save(f'{runner.work_dir}/cluster_epoch_{runner.epoch + 1}.npy',
                    new_labels.numpy())

    def evaluate(self, runner, new_labels: np.ndarray) -> None:
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

    def set_reweight(self,
                     runner,
                     labels: Optional[np.ndarray] = None,
                     reweight_pow: float = 0.5):
        """Loss re-weighting.

        Re-weighting the loss according to the number of samples in each class.

        Args:
            runner (mmengine.Runner): mmengine Runner.
            labels (numpy.ndarray): Label assignments.
            reweight_pow (float, optional): The power of re-weighting. Defaults
                to 0.5.
        """
        if labels is None:
            if self.memory_bank.label_bank.is_cuda:
                labels = self.memory_bank.label_bank.cpu().numpy()
            else:
                labels = self.memory_bank.label_bank.numpy()
        histogram = np.bincount(
            labels, minlength=self.num_classes).astype(np.float32)
        inv_histogram = (1. / (histogram + 1e-10))**reweight_pow
        weight = inv_histogram / inv_histogram.sum()
        runner.model.module.loss_weight.copy_(torch.from_numpy(weight))
