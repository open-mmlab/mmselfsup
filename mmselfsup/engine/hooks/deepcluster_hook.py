# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from mmengine.dist import is_distributed
from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmselfsup.models.utils import Extractor
from mmselfsup.registry import HOOKS
from mmselfsup.utils import clustering as _clustering


@HOOKS.register_module()
class DeepClusterHook(Hook):
    """Hook for DeepCluster.

    This hook includes the global clustering process in DC.

    Args:
        extractor (dict): Config dict for feature extraction.
        clustering (dict): Config dict that specifies the clustering algorithm.
        unif_sampling (bool): Whether to apply uniform sampling.
        reweight (bool): Whether to apply loss re-weighting.
        reweight_pow (float): The power of re-weighting.
        init_memory (bool): Whether to initialize memory banks used in ODC.
            Defaults to False.
        initial (bool): Whether to call the hook initially. Defaults to True.
        interval (int): Frequency of epochs to call the hook. Defaults to 1.
        seed (int, optional): Random seed. Defaults to None.
    """

    def __init__(
            self,
            extract_dataloader: dict,
            clustering: dict,
            unif_sampling: bool,
            reweight: bool,
            reweight_pow: float,
            init_memory: bool = False,  # for ODC
            initial: bool = True,
            interval: int = 1,
            seed: Optional[int] = None) -> None:
        self.dist_mode = is_distributed()
        self.extractor = Extractor(
            extract_dataloader=extract_dataloader,
            seed=seed,
            dist_mode=self.dist_mode,
            pool_cfg=None)
        self.clustering_type = clustering.pop('type')
        self.clustering_cfg = clustering
        self.unif_sampling = unif_sampling
        self.reweight = reweight
        self.reweight_pow = reweight_pow
        self.init_memory = init_memory
        self.initial = initial
        self.interval = interval

    def before_train(self, runner) -> None:
        """Run cluster before training."""
        self.data_loader = runner.train_dataloader
        if self.initial:
            self.deepcluster(runner)

    def after_train_epoch(self, runner) -> None:
        """Run cluster after indicated epoch."""
        if not self.every_n_epochs(runner, self.interval):
            return
        self.deepcluster(runner)

    def deepcluster(self, runner) -> None:
        """Call cluster algorithm."""
        # step 1: get features
        runner.model.eval()
        features = self.extractor(runner.model.module)['feat']
        runner.model.train()

        # step 2: get labels
        if not self.dist_mode or (self.dist_mode and runner.rank == 0):
            clustering_algo = _clustering.__dict__[self.clustering_type](
                **self.clustering_cfg)
            # Features are normalized during clustering
            clustering_algo.cluster(features, verbose=True)
            assert isinstance(clustering_algo.labels, np.ndarray)
            new_labels = clustering_algo.labels.astype(np.int64)
            np.save(f'{runner.work_dir}/cluster_epoch_{runner.epoch}.npy',
                    new_labels)
            self.evaluate(runner, new_labels)
        else:
            new_labels = np.zeros((len(self.data_loader.dataset), ),
                                  dtype=np.int64)

        if self.dist_mode:
            new_labels_tensor = torch.from_numpy(new_labels).cuda()
            dist.broadcast(new_labels_tensor, 0)
            new_labels = new_labels_tensor.cpu().numpy()
        new_labels_list = list(new_labels)

        # step 3: assign new labels
        self.data_loader.dataset.assign_labels(new_labels_list)

        # step 4 (a): set uniform sampler
        if self.unif_sampling:
            self.data_loader.sampler.set_uniform_indices(
                new_labels_list, self.clustering_cfg.k)

        # step 4 (b): set loss reweight
        if self.reweight:
            self.set_reweight(runner, new_labels, self.reweight_pow)

        # step 5: randomize classifier
        runner.model.module.head._is_init = False
        runner.model.module.head.init_weights()
        if self.dist_mode:
            for p in runner.model.module.head.state_dict().values():
                dist.broadcast(p, 0)

        # step 6: init memory for ODC
        if self.init_memory:
            runner.model.module.memory_bank.init_memory(features, new_labels)

    def evaluate(self, runner, new_labels: np.ndarray) -> None:
        """Evaluate with labels histogram."""
        histogram = np.bincount(new_labels, minlength=self.clustering_cfg.k)
        empty_cls = (histogram == 0).sum()
        minimal_cls_size, maximal_cls_size = histogram.min(), histogram.max()
        if runner.rank == 0:
            print_log(
                f'empty_num: {empty_cls.item()}\t'
                f'min_cluster: {minimal_cls_size.item()}\t'
                f'max_cluster:{maximal_cls_size.item()}',
                logger='current')

    def set_reweight(self,
                     runner,
                     labels: np.ndarray,
                     reweight_pow: float = 0.5):
        """Loss re-weighting.

        Re-weighting the loss according to the number of samples in each class.

        Args:
            runner (mmengine.Runner): mmengine Runner.
            labels (numpy.ndarray): Label assignments.
            reweight_pow (float, optional): The power of re-weighting. Defaults
                to 0.5.
        """
        histogram = np.bincount(
            labels,
            minlength=runner.model.module.memory_bank.num_classes).astype(
                np.float32)
        inv_histogram = (1. / (histogram + 1e-10))**reweight_pow
        weight = inv_histogram / inv_histogram.sum()
        runner.model.module.loss_weight.copy_(torch.from_numpy(weight))
