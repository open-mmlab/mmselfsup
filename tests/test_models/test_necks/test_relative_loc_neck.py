# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.necks import RelativeLocNeck


def test_relative_loc_neck():
    neck = RelativeLocNeck(16, 32)
    assert neck.fc.in_features == 32
    assert neck.fc.out_features == 32
    assert neck.bn.num_features == 32

    # test neck with avgpool
    fake_in = torch.rand((32, 32, 5, 5))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 32])

    # test neck without avgpool
    neck = RelativeLocNeck(16, 32, with_avg_pool=False)
    fake_in = torch.rand((32, 32))
    fake_out = neck.forward([fake_in])
    assert fake_out[0].shape == torch.Size([32, 32])
