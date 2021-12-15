# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.utils import Sobel


def test_sobel():
    sobel_layer = Sobel()
    fake_input = torch.rand((1, 3, 224, 224))
    fake_res = sobel_layer(fake_input)
    assert fake_res.shape == (1, 2, 224, 224)

    for p in sobel_layer.sobel.parameters():
        assert p.requires_grad is False
