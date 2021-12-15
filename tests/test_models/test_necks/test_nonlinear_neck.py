# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.necks import NonLinearNeck


def test_nonlinear_neck():
    # test neck arch
    neck = NonLinearNeck(16, 32, 16, norm_cfg=dict(type='BN1d'))
    assert neck.fc0.in_features == 16
    assert neck.fc0.out_features == 32
    assert neck.bn0.num_features == 32
    fc = getattr(neck, neck.fc_names[-1])
    assert fc.out_features == 16

    # test neck with avgpool
    fake_in = torch.rand((32, 16, 5, 5))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 16])

    # test neck without avgpool
    neck = NonLinearNeck(
        16, 32, 16, with_avg_pool=False, norm_cfg=dict(type='BN1d'))
    fake_in = torch.rand((32, 16))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 16])
