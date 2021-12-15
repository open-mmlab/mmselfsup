# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmselfsup.models.necks import SwAVNeck


def test_swav_neck():
    neck = SwAVNeck(16, 32, 16, norm_cfg=dict(type='BN1d'))
    assert isinstance(neck.projection_neck, (nn.Module, nn.Sequential))

    # test neck with avgpool
    fake_in = [[torch.rand((32, 16, 5, 5))], [torch.rand((32, 16, 5, 5))],
               [torch.rand((32, 16, 3, 3))]]
    fake_out = neck.forward(fake_in)
    assert fake_out[0].shape == torch.Size([32 * len(fake_in), 16])
