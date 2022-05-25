# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, get_dist_info

from ..builder import HEADS, build_neck


@HEADS.register_module()
class LatentPredictHead(BaseModule):
    """Head for latent feature prediction.

    This head builds a predictor, which can be any registered neck component.
    For example, BYOL and SimSiam call this head and build NonLinearNeck.
    It also implements similarity loss between two forward features.

    Args:
        predictor (Dict): Config dict for the predictor.
    """

    def __init__(self, predictor: Dict) -> None:
        super().__init__()
        self.predictor = build_neck(predictor)

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward head.

        Args:
            input (torch.Tensor): NxC input features.
            target (torch.Tensor): NxC target features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted features and
                target features.
        """
        pred = self.predictor([input])[0]
        target = target.detach()

        return pred, target


@HEADS.register_module()
class LatentClsHead(BaseModule):
    """Head for latent feature classification.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes.
        init_cfg (Optional[Union[Dict, List[Dict]]], optional): Initialization
            config dict.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
            type='Normal',
            std=0.01,
            layer='Linear',
        )
    ) -> None:
        super(LatentClsHead, self).__init__(init_cfg)
        self.predictor = nn.Linear(in_channels, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Dict:
        """Forward head.

        Args:
            input (torch.Tensor): NxC input features.
            target (torch.Tensor): NxC target features.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(input)
        with torch.no_grad():
            label = torch.argmax(self.predictor(target), dim=1).detach()
        loss = self.criterion(pred, label)
        return dict(loss=loss)


@HEADS.register_module()
class LatentCrossCorrelationHead(BaseModule):
    """Head for latent feature cross correlation. Part of the code is borrowed
    from:
    `https://github.com/facebookresearch/barlowtwins/blob/main/main.py>`_.

    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        _, self.world_size = get_dist_info()
        self.bn = nn.BatchNorm1d(in_channels, affine=False)

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Forward head.

        Args:
            input (torch.Tensor): NxC input features.
            target (torch.Tensor): NxC target features.

        Returns:
            torch.Tensor: The cross correlation matrix.
        """
        # cross-correlation matrix
        cross_correlation_matrix = self.bn(input).T @ self.bn(target)
        cross_correlation_matrix.div_(input.size(0) * self.world_size)

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(cross_correlation_matrix)

        return cross_correlation_matrix
