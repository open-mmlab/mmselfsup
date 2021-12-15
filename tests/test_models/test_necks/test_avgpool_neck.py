# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.necks import AvgPool2dNeck


def test_avgpool2d_neck():
    fake_in = [torch.randn((2, 3, 8, 8))]

    # test default
    neck = AvgPool2dNeck()
    fake_out = neck(fake_in)
    assert fake_out[0].shape == (2, 3, 1, 1)

    # test custom
    neck = AvgPool2dNeck(2)
    fake_out = neck(fake_in)
    assert fake_out[0].shape == (2, 3, 2, 2)

    # test custom
    neck = AvgPool2dNeck((1, 2))
    fake_out = neck(fake_in)
    assert fake_out[0].shape == (2, 3, 1, 2)
