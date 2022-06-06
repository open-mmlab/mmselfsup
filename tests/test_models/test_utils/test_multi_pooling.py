# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.models.utils import MultiPooling


def test_multi_pooling():
    # adaptive
    layer = MultiPooling(pool_type='adaptive', in_indices=(0, 1, 2))
    fake_in = [
        torch.rand((1, 32, 112, 112)),
        torch.rand((1, 64, 56, 56)),
        torch.rand((1, 128, 28, 28)),
    ]
    res = layer(fake_in)
    assert res[0].shape == (1, 32, 12, 12)
    assert res[1].shape == (1, 64, 6, 6)
    assert res[2].shape == (1, 128, 4, 4)

    # specified
    layer = MultiPooling(pool_type='specified', in_indices=(0, 1, 2))
    fake_in = [
        torch.rand((1, 32, 112, 112)),
        torch.rand((1, 64, 56, 56)),
        torch.rand((1, 128, 28, 28)),
    ]
    res = layer(fake_in)
    assert res[0].shape == (1, 32, 12, 12)
    assert res[1].shape == (1, 64, 6, 6)
    assert res[2].shape == (1, 128, 4, 4)

    with pytest.raises(AssertionError):
        layer = MultiPooling(pool_type='other')

    with pytest.raises(AssertionError):
        layer = MultiPooling(backbone='resnet101')
