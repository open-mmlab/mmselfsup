# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import HEADS, build_neck


@HEADS.register_module()
class LatentPredictHead(BaseModule):
    """Head for latent feature prediction.

    This head builds a predictor, which can be any registered neck component.
    For example, BYOL and SimSiam call this head and build NonLinearNeck.
    It also implements similarity loss between two forward features.

    Args:
        predictor (dict): Config dict for module of predictor.
    """

    def __init__(self, predictor):
        super(LatentPredictHead, self).__init__()
        self.predictor = build_neck(predictor)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor([input])[0]
        target = target.detach()

        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = -(pred_norm * target_norm).sum(dim=1).mean()
        return dict(loss=loss)


@HEADS.register_module
class LatentClsHead(BaseModule):
    """Head for latent feature classification.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 init_cfg=dict(type='Normal', std=0.01, layer='Linear')):
        super(LatentClsHead, self).__init__(init_cfg)
        self.predictor = nn.Linear(in_channels, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(input)
        with torch.no_grad():
            label = torch.argmax(self.predictor(target), dim=1).detach()
        loss = self.criterion(pred, label)
        return dict(loss=loss)


@HEADS.register_module()
class LatentCrossCorrelationHead(BaseModule):
    """Head for latent feature cross correlation.

    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels, lambd=0.0051):
        super(LatentCrossCorrelationHead, self).__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(in_channels, affine=False)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of
            # a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        c = self.bn(input).T @ self.bn(target)
        c.div_(input.size(0))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return dict(loss=loss)
