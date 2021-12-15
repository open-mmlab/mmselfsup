# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule


def accuracy(pred, target, topk=1):
    """Compute accuracy of predictions.

    Args:
        pred (Tensor): The output of the model.
        target (Tensor): The labels of data.
        topk (int | list[int]): Top-k metric selection. Defaults to 1.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.contiguous().view(1,
                                                     -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(
            0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


class Accuracy(BaseModule):
    """Implementation of accuracy computation."""

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)
