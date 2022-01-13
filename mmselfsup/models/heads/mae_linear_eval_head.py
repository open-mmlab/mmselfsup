# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule
from timm.models.layers import trunc_normal_
from torch import nn

from ..builder import HEADS


@HEADS.register_module()
class MAELinearEvalHead(BaseModule):
    """Linear evaluation head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000):
        super(MAELinearEvalHead, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.bn = nn.BatchNorm1d(embed_dim, affine=False)
        self.criterion = nn.CrossEntropyLoss()
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, x):
        """"Get the logits."""
        x = self.bn(x)
        outputs = self.head(x)

        return outputs

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs, labels)

        return losses
