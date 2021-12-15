# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock

import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_runner
from torch.utils.data import Dataset

from mmselfsup.core.hooks import DeepClusterHook
from mmselfsup.core.optimizer import build_optimizer
from mmselfsup.models.algorithms import DeepCluster

num_classes = 10
with_sobel = True,
backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=2,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(type='AvgPool2dNeck')
head = dict(
    type='ClsHead',
    with_avg_pool=False,  # already has avgpool in the neck
    in_channels=2048,
    num_classes=num_classes)


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(img=torch.randn((3, 224, 224)), img_metas=dict())
        return results

    def __len__(self):
        return 10


def test_deepcluster_hook():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))

    alg = DeepCluster(
        backbone=backbone, with_sobel=with_sobel, neck=neck, head=head)
    extractor = dict(
        dataset=test_dataset,
        imgs_per_gpu=1,
        workers_per_gpu=0,
        persistent_workers=False)

    runner_cfg = dict(type='EpochBasedRunner', max_epochs=3)
    optim_cfg = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
    lr_config = dict(policy='CosineAnnealing', min_lr=0.)

    # test DeepClusterHook
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MMDataParallel(alg)
        optimizer = build_optimizer(model, optim_cfg)
        deepcluster_hook = DeepClusterHook(
            extractor=extractor,
            clustering=dict(type='Kmeans', k=num_classes, pca_dim=16),
            unif_sampling=True,
            reweight=False,
            reweight_pow=0.5,
            initial=True,
            interval=1,
            dist_mode=False)
        runner = build_runner(
            runner_cfg,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logging.getLogger()))
        runner.register_training_hooks(lr_config)
        runner.register_hook(deepcluster_hook)
        assert deepcluster_hook.clustering_type == 'Kmeans'
