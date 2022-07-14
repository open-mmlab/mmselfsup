# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.data import InstanceData

from mmselfsup.core import SelfSupDataSample
from mmselfsup.models.algorithms import DeepCluster

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
    loss=dict(type='mmcls.CrossEntropyLoss'),
    with_avg_pool=False,  # already has avgpool in the neck
    in_channels=512,
    num_classes=num_classes)


@pytest.mark.skipif(
    not torch.cuda.is_available() or platform.system() == 'Windows',
    reason='CUDA is not available or Windows mem limit')
def test_deepcluster():
    data_preprocessor = {
        'mean': (123.675, 116.28, 103.53),
        'std': (58.395, 57.12, 57.375),
        'bgr_to_rgb': True
    }

    alg = DeepCluster(
        backbone=backbone,
        neck=neck,
        head=head,
        data_preprocessor=copy.deepcopy(data_preprocessor))
    assert alg.num_classes == num_classes
    assert hasattr(alg, 'sobel_layer')
    assert hasattr(alg, 'neck')
    assert hasattr(alg, 'head')

    fake_data_sample = SelfSupDataSample()
    fake_label = InstanceData(label=torch.tensor([1]))
    fake_data_sample.pseudo_label = fake_label
    fake_data = [{
        'inputs': [torch.randn(3, 224, 224)],
        'data_sample': fake_data_sample
    } for _ in range(2)]

    fake_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_loss = alg(fake_inputs, fake_data_samples, mode='loss')
    assert fake_loss['loss'] > 0

    # test extract
    fake_feats = alg(fake_inputs, fake_data_samples, mode='tensor')
    assert fake_feats[0].size() == torch.Size([2, 512, 7, 7])
