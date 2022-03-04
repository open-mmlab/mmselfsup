# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import NPID

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='LinearNeck', in_channels=2048, out_channels=4, with_avg_pool=True)
head = dict(type='ContrastiveHead', temperature=0.07)
memory_bank = dict(type='SimpleMemory', length=8, feat_dim=4, momentum=0.5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or platform.system() == 'Windows',
    reason='CUDA is not available or Windows mem limit')
def test_npid():
    with pytest.raises(AssertionError):
        alg = NPID(backbone=backbone, neck=neck, head=head, memory_bank=None)
    with pytest.raises(AssertionError):
        alg = NPID(
            backbone=backbone, neck=neck, head=None, memory_bank=memory_bank)

    alg = NPID(
        backbone=backbone, neck=neck, head=head, memory_bank=memory_bank)
    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.extract_feat(fake_input)
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
