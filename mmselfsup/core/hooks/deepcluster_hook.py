# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook
from mmcv.utils import print_log

from mmselfsup.utils import Extractor
from mmselfsup.utils import clustering as _clustering
from mmselfsup.utils import get_root_logger


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
        dist_mode (bool): Use distributed training or not. Defaults to True.
        data_loaders (DataLoader): A PyTorch dataloader. Defaults to None.
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

        logger = get_root_logger()
        if 'imgs_per_gpu' in extractor:
            logger.warning('"imgs_per_gpu" is deprecated. '
                           'Please use "samples_per_gpu" instead')
            if 'samples_per_gpu' in extractor:
                logger.warning(
                    f'Got "imgs_per_gpu"={extractor["imgs_per_gpu"]} and '
                    f'"samples_per_gpu"={extractor["samples_per_gpu"]}, '
                    f'"imgs_per_gpu"={extractor["imgs_per_gpu"]} is used in '
                    f'this experiments')
            else:
                logger.warning(
                    'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                    f'{extractor["imgs_per_gpu"]} in this experiments')
            extractor['samples_per_gpu'] = extractor['imgs_per_gpu']

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
            np.save(f'{runner.work_dir}/cluster_epoch_{runner.epoch}.npy',
                    new_labels)
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
        runner.model.module.head._is_init = False
        runner.model.module.head.init_weights()
        if self.dist_mode:
            for p in runner.model.module.head.state_dict().values():
                dist.broadcast(p, 0)

        # step 6: init memory for ODC
        if self.init_memory:
            runner.model.module.memory_bank.init_memory(features, new_labels)

    def evaluate(self, runner, new_labels):
        histogram = np.bincount(new_labels, minlength=self.clustering_cfg.k)
        empty_cls = (histogram == 0).sum()
        minimal_cls_size, maximal_cls_size = histogram.min(), histogram.max()
        if runner.rank == 0:
            print_log(
                f'empty_num: {empty_cls.item()}\t'
                f'min_cluster: {minimal_cls_size.item()}\t'
                f'max_cluster:{maximal_cls_size.item()}',
                logger='root')
