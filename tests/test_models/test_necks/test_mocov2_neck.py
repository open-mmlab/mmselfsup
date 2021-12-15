# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmselfsup.models.necks import MoCoV2Neck


def test_mocov2_neck():
    neck = MoCoV2Neck(16, 32, 16)
    assert isinstance(neck.mlp, nn.Sequential)
    assert neck.mlp[0].in_features == 16
    assert neck.mlp[2].in_features == 32
    assert neck.mlp[2].out_features == 16

    # test neck with avgpool
    fake_in = torch.rand((32, 16, 5, 5))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 16])

    # test neck without avgpool
    neck = MoCoV2Neck(16, 32, 16, with_avg_pool=False)
    fake_in = torch.rand((32, 16))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 16])
