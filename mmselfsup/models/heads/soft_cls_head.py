# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from timm.models.layers import trunc_normal_

from ..builder import HEADS


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


@HEADS.register_module()
class SoftClsHead(BaseModule):
    """Head for pixel_level reconstruction.

    The MSE loss is implemented in this head and is used in generative methods,
    e.g. MAE
    """

    def __init__(self, embed_dim, num_classes, init_scale):
        super(SoftClsHead, self).__init__()
        self.criterion = SoftTargetCrossEntropy()
        self.head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def forward(self, x):
        """"Get the logits."""
        outputs = self.head(x)

        return outputs

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs, labels)

        return losses
