# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import RelativeLoc

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
neck = dict(
    type='RelativeLocNeck',
    in_channels=2048,
    out_channels=4,
    with_avg_pool=True)
head = dict(type='ClsHead', with_avg_pool=False, in_channels=4, num_classes=8)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_relative_loc():
    with pytest.raises(AssertionError):
        alg = RelativeLoc(backbone=backbone, neck=None, head=head)
    with pytest.raises(AssertionError):
        alg = RelativeLoc(backbone=backbone, neck=neck, head=None)

    alg = RelativeLoc(backbone=backbone, neck=neck, head=head)

    with pytest.raises(AssertionError):
        fake_input = torch.randn((2, 8, 6, 224, 224))
        patch_labels = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
        alg.forward(fake_input, patch_labels)

    # train
    fake_input = torch.randn((2, 8, 6, 224, 224))
    patch_labels = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                     [0, 1, 2, 3, 4, 5, 6, 7]])
    fake_out = alg.forward(fake_input, patch_labels)
    assert fake_out['loss'].item() > 0

    # test
    fake_input = torch.randn((2, 8, 6, 224, 224))
    patch_labels = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                     [0, 1, 2, 3, 4, 5, 6, 7]])
    fake_out = alg.forward(fake_input, patch_labels, mode='test')
    assert 'head4' in fake_out

    # extract
    fake_input = torch.randn((16, 3, 224, 224))
    fake_backbone_out = alg.forward(fake_input, mode='extract')
    assert fake_backbone_out[0].size() == torch.Size([16, 2048, 7, 7])
