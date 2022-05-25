# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.data import LabelData

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
    with_avg_pool=False,  # already has avgpool in the neck
    in_channels=512,
    num_classes=num_classes)
loss = dict(type='mmcls.CrossEntropyLoss')
preprocess_cfg = {
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
    'to_rgb': True
}


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_deepcluster():
    with pytest.raises(AssertionError):
        alg = DeepCluster(
            backbone=backbone,
            with_sobel=with_sobel,
            neck=neck,
            head=None,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    alg = DeepCluster(
        backbone=backbone,
        with_sobel=with_sobel,
        neck=neck,
        head=head,
        loss=loss,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))
    assert alg.num_classes == num_classes
    assert hasattr(alg, 'sobel_layer')
    assert hasattr(alg, 'neck')
    assert hasattr(alg, 'head')

    fake_data_sample = SelfSupDataSample()
    fake_label = LabelData(value=torch.tensor([1]))
    fake_data_sample.pred_label = fake_label
    fake_input = [{
        'inputs': [torch.randn(3, 224, 224)],
        'data_sample': fake_data_sample
    } for _ in range(2)]

    fake_out = alg(fake_input, return_loss=False)
    assert hasattr(fake_out[0].prediction, 'head0')
    assert fake_out[0].prediction.head0.size() == torch.Size([num_classes])

    fake_out = alg(fake_input, return_loss=True)
    assert fake_out['loss'].item() > 0
