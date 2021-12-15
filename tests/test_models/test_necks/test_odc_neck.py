# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.necks import ODCNeck


def test_odc_neck():
    neck = ODCNeck(16, 32, 16, norm_cfg=dict(type='BN1d'))
    assert neck.fc0.in_features == 16
    assert neck.fc0.out_features == 32
    assert neck.bn0.num_features == 32
    assert neck.fc1.in_features == 32
    assert neck.fc1.out_features == 16

    # test neck with avgpool
    fake_in = torch.rand((32, 16, 5, 5))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 16])

    # test neck without avgpool
    neck = ODCNeck(16, 32, 16, with_avg_pool=False, norm_cfg=dict(type='BN1d'))
    fake_in = torch.rand((32, 16))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 16])
