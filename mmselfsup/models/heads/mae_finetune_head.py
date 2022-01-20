# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule
from torch import nn

from ..builder import HEADS


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


@HEADS.register_module()
class MAEFinetuneHead(BaseModule):
    """Fine-tuning head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000):
        super(MAEFinetuneHead, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.criterion = SoftTargetCrossEntropy()
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, x):
        """"Get the logits."""
        outputs = self.head(x)

        return outputs

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs, labels)

        return losses
