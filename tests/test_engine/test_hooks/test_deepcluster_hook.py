# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from unittest import TestCase

import torch
from mmengine.structures import LabelData
from torch.utils.data import Dataset

from mmselfsup.engine.hooks import DeepClusterHook
from mmselfsup.structures import SelfSupDataSample

num_classes = 5
with_sobel = True,
backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=2,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(type='AvgPool2dNeck')
head = dict(
    type='ClsHead',
    with_avg_pool=False,  # already has avgpool in the neck
    in_channels=512,
    num_classes=num_classes)
loss = dict(type='mmcls.CrossEntropyLoss')


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        data_sample = SelfSupDataSample()
        gt_label = LabelData(value=self.label[index])
        setattr(data_sample, 'gt_label', gt_label)
        return dict(inputs=self.data[index], data_sample=data_sample)


class TestDeepClusterHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_deepcluster_hook(self):
        dummy_dataset = DummyDataset()

        extract_dataloader = dict(
            dataset=dummy_dataset,
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=1,
            num_workers=0,
            persistent_workers=False)
        deepcluster_hook = DeepClusterHook(
            extract_dataloader=extract_dataloader,
            clustering=dict(type='Kmeans', k=num_classes, pca_dim=16),
            unif_sampling=True,
            reweight=False,
            reweight_pow=0.5,
            initial=True,
            interval=1)

        # test DeepClusterHook
        assert deepcluster_hook.clustering_type == 'Kmeans'
