# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from ..builder import HEADS

HEADS.register_module()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


@HEADS.register_module()
class VitHead(BaseModule):
    """Head for pixel_level reconstruction.

    The MSE loss is implemented in this head and is used in generative methods,
    e.g. MAE
    """

    def __init__(self):
        super(VitHead, self).__init__()
        self.criterion = SoftTargetCrossEntropy()

    def forward(self, outputs, labels):

        losses = dict()

        losses['loss'] = self.criterion(outputs, labels)

        return losses
