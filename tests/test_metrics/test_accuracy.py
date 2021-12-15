# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.utils import Accuracy


def test_accuracy():
    pred = torch.Tensor([[0.2, 0.3, 0.5], [0.25, 0.15, 0.6], [0.9, 0.05, 0.05],
                         [0.8, 0.1, 0.1], [0.55, 0.15, 0.3]])
    target = torch.zeros(5)

    acc = Accuracy((1, 2))
    res = acc.forward(pred, target)
    assert res[0].item() == 60.
    assert res[1].item() == 80.
