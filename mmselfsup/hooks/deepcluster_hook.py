import numpy as np

from mmcv.runner import Hook

import torch
import torch.distributed as dist

from openselfsup.third_party import clustering as _clustering
from openselfsup.utils import print_log
from .registry import HOOKS
from .extractor import Extractor


@HOOKS.register_module
class DeepClusterHook(Hook):
    """Hook for DeepCluster.

    Args:
        extractor (dict): Config dict for feature extraction.
        clustering (dict): Config dict that specifies the clustering algorithm.
        unif_sampling (bool): Whether to apply uniform sampling.
        reweight (bool): Whether to apply loss re-weighting.
        reweight_pow (float): The power of re-weighting.
        init_memory (bool): Whether to initialize memory banks for ODC.
            Default: False.
        initial (bool): Whether to call the hook initially. Default: True.
        interval (int): Frequency of epochs to call the hook. Default: 1.
        dist_mode (bool): Use distributed training or not. Default: True.
        data_loaders (DataLoader): A PyTorch dataloader. Default: None.
    """

    def __init__(
            self,
            extractor,
            clustering,
            unif_sampling,
            reweight,
            reweight_pow,
            init_memory=False,  # for ODC
            initial=True,
            interval=1,
            dist_mode=True,
            data_loaders=None):
        self.extractor = Extractor(dist_mode=dist_mode, **extractor)
        self.clustering_type = clustering.pop('type')
        self.clustering_cfg = clustering
        self.unif_sampling = unif_sampling
        self.reweight = reweight
        self.reweight_pow = reweight_pow
        self.init_memory = init_memory
        self.initial = initial
        self.interval = interval
        self.dist_mode = dist_mode
        self.data_loaders = data_loaders

    def before_run(self, runner):
        if self.initial:
            self.deepcluster(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
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
            np.save(
                "{}/cluster_epoch_{}.npy".format(runner.work_dir,
                                                 runner.epoch), new_labels)
            self.evaluate(runner, new_labels)
        else:
            new_labels = np.zeros((len(self.data_loaders[0].dataset), ),
                                  dtype=np.int64)

        if self.dist_mode:
            new_labels_tensor = torch.from_numpy(new_labels).cuda()
            dist.broadcast(new_labels_tensor, 0)
            new_labels = new_labels_tensor.cpu().numpy()
        new_labels_list = list(new_labels)

        # step 3: assign new labels
        self.data_loaders[0].dataset.assign_labels(new_labels_list)

        # step 4 (a): set uniform sampler
        if self.unif_sampling:
            self.data_loaders[0].sampler.set_uniform_indices(
                new_labels_list, self.clustering_cfg.k)

        # step 4 (b): set loss reweight
        if self.reweight:
            runner.model.module.set_reweight(new_labels, self.reweight_pow)

        # step 5: randomize classifier
        runner.model.module.head.init_weights(init_linear='normal')
        if self.dist_mode:
            for p in runner.model.module.head.state_dict().values():
                dist.broadcast(p, 0)

        # step 6: init memory for ODC
        if self.init_memory:
            runner.model.module.memory_bank.init_memory(features, new_labels)

    def evaluate(self, runner, new_labels):
        hist = np.bincount(new_labels, minlength=self.clustering_cfg.k)
        empty_cls = (hist == 0).sum()
        minimal_cls_size, maximal_cls_size = hist.min(), hist.max()
        if runner.rank == 0:
            print_log(
                "empty_num: {}\tmin_cluster: {}\tmax_cluster:{}".format(
                    empty_cls.item(), minimal_cls_size.item(),
                    maximal_cls_size.item()),
                logger='root')
