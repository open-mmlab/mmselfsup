# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch

from mmselfsup.core import SelfSupDataSample
from mmselfsup.models.algorithms import NPID

backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='LinearNeck', in_channels=512, out_channels=2, with_avg_pool=True)
head = dict(type='ContrastiveHead', temperature=0.07)
loss = dict(type='mmcls.CrossEntropyLoss'),
memory_bank = dict(type='SimpleMemory', length=8, feat_dim=2, momentum=0.5)
preprocess_cfg = {
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
    'to_rgb': True
}


@pytest.mark.skipif(
    not torch.cuda.is_available() or platform.system() == 'Windows',
    reason='CUDA is not available or Windows mem limit')
def test_npid():
    with pytest.raises(AssertionError):
        alg = NPID(
            backbone=backbone,
            neck=neck,
            head=head,
            memory_bank=None,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = NPID(
            backbone=backbone,
            neck=neck,
            head=None,
            loss=loss,
            memory_bank=memory_bank,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        alg = NPID(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=None,
            memory_bank=memory_bank,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))

    alg = NPID(
        backbone=backbone,
        neck=neck,
        head=head,
        loss=loss,
        memory_bank=memory_bank,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))

    fake_data = [{
        'inputs': torch.randn((3, 224, 224)),
        'data_sample': SelfSupDataSample()
    } for _ in range(2)]

    fake_inputs, _ = alg.preprocss_data(fake_data)
    fake_backbone_out = alg.extract_feat(fake_inputs)
    assert fake_backbone_out[0].size() == torch.Size([2, 512, 7, 7])
