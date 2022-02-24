# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import RotationPred

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
head = dict(
    type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=4)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_rotation_pred():
    with pytest.raises(AssertionError):
        alg = RotationPred(backbone=backbone, head=None)

    alg = RotationPred(backbone=backbone, head=head)

    with pytest.raises(AssertionError):
        fake_input = torch.randn((2, 4, 3, 224, 224))
        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        alg.forward(fake_input, rotation_labels)

    # train
    fake_input = torch.randn((2, 4, 3, 224, 224))
    rotation_labels = torch.LongTensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    fake_out = alg.forward(fake_input, rotation_labels)
    assert fake_out['loss'].item() > 0

    # test
    fake_input = torch.randn((2, 4, 3, 224, 224))
    rotation_labels = torch.LongTensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    fake_out = alg.forward(fake_input, rotation_labels, mode='test')
    assert 'head4' in fake_out

    # extract
    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.forward(fake_input, mode='extract')
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
