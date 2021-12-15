# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmselfsup.models.necks import LinearNeck


def test_linear_neck():
    neck = LinearNeck(16, 32, with_avg_pool=True)
    assert isinstance(neck.avgpool, nn.Module)
    assert neck.fc.in_features == 16
    assert neck.fc.out_features == 32

    # test neck with avgpool
    fake_in = torch.rand((32, 16, 5, 5))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 32])

    # test neck without avgpool
    neck = LinearNeck(16, 32, with_avg_pool=False)
    fake_in = torch.rand((32, 16))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 32])
