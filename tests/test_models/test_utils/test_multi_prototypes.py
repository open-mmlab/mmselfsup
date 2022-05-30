# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmselfsup.models.utils import MultiPrototypes


def test_multi_prototypes():
    with pytest.raises(AssertionError):
        layer = MultiPrototypes(output_dim=16, num_prototypes=2)

    layer = MultiPrototypes(output_dim=16, num_prototypes=[3, 4, 5])
    assert isinstance(getattr(layer, 'prototypes0'), nn.Module)
    assert isinstance(getattr(layer, 'prototypes1'), nn.Module)
    assert isinstance(getattr(layer, 'prototypes2'), nn.Module)

    fake_in = torch.rand((32, 16))
    res = layer.forward(fake_in)
    assert len(res) == 3
    assert res[0].shape == (32, 3)
    assert res[1].shape == (32, 4)
    assert res[2].shape == (32, 5)
