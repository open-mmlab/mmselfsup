# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook
from mmcv.utils import print_log

from mmselfsup.utils import Extractor
from mmselfsup.utils import clustering as _clustering


@HOOKS.register_module()
class InterCLRHook(Hook):
    """Hook for InterCLR.

    This hook includes the clustering process in InterCLR.

    Args:
        extractor (dict): Config dict for feature extraction.
        clustering (dict): Config dict that specifies the clustering algorithm.
        centroids_update_interval (int): Frequency of iterations to
            update centroids.
        deal_with_small_clusters_interval (int): Frequency of iterations to
            deal with small clusters.
        evaluate_interval (int): Frequency of iterations to evaluate clusters.
        warmup_epochs (int, optional): The number of warmup epochs to set
            ``intra_loss_weight=1`` and ``inter_loss_weight=0``. Defaults to 0.
        init_memory (bool): Whether to initialize memory banks used in online
            labels. Defaults to True.
        initial (bool): Whether to call the hook initially. Defaults to True.
        online_labels (bool): Whether to use online labels. Defaults to True.
        interval (int): Frequency of epochs to call the hook. Defaults to 1.
        dist_mode (bool): Use distributed training or not. Defaults to True.
        data_loaders (DataLoader): A PyTorch dataloader. Defaults to None.
    """

    def __init__(
            self,
            extractor,
            clustering,
            centroids_update_interval,
            deal_with_small_clusters_interval,
            evaluate_interval,
            warmup_epochs=0,
            init_memory=True,
            initial=True,
            online_labels=True,
            interval=1,  # same as the checkpoint interval
            dist_mode=True,
            data_loaders=None):
        assert dist_mode, 'non-dist mode is not implemented'
        self.extractor = Extractor(dist_mode=dist_mode, **extractor)
        self.clustering_type = clustering.pop('type')
        self.clustering_cfg = clustering
        self.centroids_update_interval = centroids_update_interval
        self.deal_with_small_clusters_interval = \
            deal_with_small_clusters_interval
        self.evaluate_interval = evaluate_interval
        self.warmup_epochs = warmup_epochs
        self.init_memory = init_memory
        self.initial = initial
        self.online_labels = online_labels
        self.interval = interval
        self.dist_mode = dist_mode
        self.data_loaders = data_loaders

    def before_run(self, runner):
        assert hasattr(runner.model.module, 'intra_loss_weight'), \
            "The runner must have attribute \"intra_loss_weight\" in InterCLR."
        assert hasattr(runner.model.module, 'inter_loss_weight'), \
            "The runner must have attribute \"inter_loss_weight\" in InterCLR."
        self.intra_loss_weight = runner.model.module.intra_loss_weight
        self.inter_loss_weight = runner.model.module.inter_loss_weight
        if self.initial:
            if runner.epoch > 0 and self.online_labels:
                if runner.rank == 0:
                    print(f'Resuming memory banks from epoch {runner.epoch}')
                    features = np.load(
                        f'{runner.work_dir}/feature_epoch_{runner.epoch}.npy')
                else:
                    features = None
                loaded_labels = np.load(
                    f'{runner.work_dir}/cluster_epoch_{runner.epoch}.npy')
                runner.model.module.memory_bank.init_memory(
                    features, loaded_labels)
                return

            self.deepcluster(runner)

    def before_train_epoch(self, runner):
        cur_epoch = runner.epoch
        if cur_epoch >= self.warmup_epochs:
            runner.model.module.intra_loss_weight = self.intra_loss_weight
            runner.model.module.inter_loss_weight = self.inter_loss_weight
        else:
            runner.model.module.intra_loss_weight = 1.
            runner.model.module.inter_loss_weight = 0.

    def after_train_iter(self, runner):
        if not self.online_labels:
            return
        # centroids update
        if self.every_n_iters(runner, self.centroids_update_interval):
            runner.model.module.memory_bank.update_centroids_memory()

        # deal with small clusters
        if self.every_n_iters(runner, self.deal_with_small_clusters_interval):
            runner.model.module.memory_bank.deal_with_small_clusters()

        # evaluate
        if self.every_n_iters(runner, self.evaluate_interval):
            new_labels = runner.model.module.memory_bank.label_bank
            if new_labels.is_cuda:
                new_labels = new_labels.cpu()
            self.evaluate(runner, new_labels.numpy())

    def after_train_epoch(self, runner):
        if self.online_labels:  # online labels
            # save cluster
            if self.every_n_epochs(runner, self.interval) and runner.rank == 0:
                features = runner.model.module.memory_bank.feature_bank
                new_labels = runner.model.module.memory_bank.label_bank
                if new_labels.is_cuda:
                    new_labels = new_labels.cpu()
                np.save(
                    f'{runner.work_dir}/feature_epoch_{runner.epoch + 1}.npy',
                    features.cpu().numpy())
                np.save(
                    f'{runner.work_dir}/cluster_epoch_{runner.epoch + 1}.npy',
                    new_labels.numpy())
        else:  # offline labels
            if self.every_n_epochs(runner, self.interval):
                self.deepcluster(runner)

    def deepcluster(self, runner):
        # step 1: get features
        runner.model.eval()
        features = self.extractor(runner)
        runner.model.train()

        # step 2: get labels
        if not self.dist_mode or (self.dist_mode and runner.rank == 0):
            clustering_algo = _clustering.__dict__[self.clustering_type](
                **self.clustering_cfg)
            # Features are normalized during clustering
            clustering_algo.cluster(features, verbose=True)
            assert isinstance(clustering_algo.labels, np.ndarray)
            new_labels = clustering_algo.labels.astype(np.int64)
            if self.init_memory:
                np.save(f'{runner.work_dir}/cluster_epoch_{runner.epoch}.npy',
                        new_labels)
            else:
                np.save(
                    f'{runner.work_dir}/cluster_epoch_{runner.epoch + 1}.npy',
                    new_labels)
            self.evaluate(runner, new_labels)
        else:
            new_labels = np.zeros((len(self.data_loaders[0].dataset), ),
                                  dtype=np.int64)

        if self.dist_mode:
            new_labels_tensor = torch.from_numpy(new_labels).cuda()
            dist.broadcast(new_labels_tensor, 0)
            new_labels = new_labels_tensor.cpu().numpy()

        # step 3 (optional): assign offline labels
        if not (self.online_labels or self.init_memory):
            runner.model.module.memory_bank.assign_label(new_labels)

        # step 4 (before run): initialize memory
        if self.init_memory:
            runner.model.module.memory_bank.init_memory(features, new_labels)
            self.init_memory = False

    def evaluate(self, runner, new_labels):
        histogram = np.bincount(
            new_labels, minlength=runner.model.module.memory_bank.num_classes)
        empty_cls = (histogram == 0).sum()
        minimal_cls_size, maximal_cls_size = histogram.min(), histogram.max()
        if runner.rank == 0:
            print_log(
                f'empty_num: {empty_cls.item()}\t'
                f'min_cluster: {minimal_cls_size.item()}\t'
                f'max_cluster: {maximal_cls_size.item()}',
                logger='mmselfsup')
