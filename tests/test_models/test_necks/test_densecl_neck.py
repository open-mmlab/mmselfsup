# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmselfsup.models.necks import DenseCLNeck


def test_densecl_neck():
    neck = DenseCLNeck(16, 32, 16)
    assert isinstance(neck.mlp, nn.Sequential)
    assert isinstance(neck.mlp2, nn.Sequential)
    assert neck.mlp[0].in_features == 16
    assert neck.mlp[2].in_features == 32
    assert neck.mlp[2].out_features == 16
    assert neck.mlp2[0].in_channels == 16
    assert neck.mlp2[2].in_channels == 32
    assert neck.mlp2[2].out_channels == 16

    # test neck when num_grid is None
    fake_in = torch.rand((32, 16, 5, 5))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 16])
    assert fake_out[1].shape == torch.Size([32, 16, 25])
    assert fake_out[2].shape == torch.Size([32, 16])

    # test neck when num_grid is not None
    neck = DenseCLNeck(16, 32, 16, num_grid=3)
    fake_in = torch.rand((32, 16, 5, 5))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 16])
    assert fake_out[1].shape == torch.Size([32, 16, 9])
    assert fake_out[2].shape == torch.Size([32, 16])
