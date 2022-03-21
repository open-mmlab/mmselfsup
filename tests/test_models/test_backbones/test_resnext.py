# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.models.backbones import ResNeXt
from mmselfsup.models.backbones.resnext import Bottleneck as BottleneckX


def test_resnext():
    with pytest.raises(KeyError):
        # ResNeXt depth should be in [50, 101, 152]
        ResNeXt(depth=18)

    # Test ResNeXt with group 32, width_per_group 4
    model = ResNeXt(
        depth=50, groups=32, width_per_group=4, out_indices=(0, 1, 2, 3, 4))
    for m in model.modules():
        if isinstance(m, BottleneckX):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size([1, 64, 112, 112])
    assert feat[1].shape == torch.Size([1, 256, 56, 56])
    assert feat[2].shape == torch.Size([1, 512, 28, 28])
    assert feat[3].shape == torch.Size([1, 1024, 14, 14])
    assert feat[4].shape == torch.Size([1, 2048, 7, 7])

    # Test ResNeXt with group 32, width_per_group 4 and layers 3 out forward
    model = ResNeXt(depth=50, groups=32, width_per_group=4, out_indices=(4, ))
    for m in model.modules():
        if isinstance(m, BottleneckX):
            assert m.conv2.groups == 32
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 2048, 7, 7])
