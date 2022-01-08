# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.backbones import MAEPretrainViT


def test_vit():
    self = MAEPretrainViT()
    self.eval()
    inputs = torch.rand(1, 3, 224, 224)
    level_outputs = self.forward(inputs)
    assert tuple(level_outputs.shape) == (1, 50, 1024)
