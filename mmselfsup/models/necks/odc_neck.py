# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class ODCNeck(BaseModule):
    """The non-linear neck of ODC: fc-bn-relu-dropout-fc-relu.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        with_avg_pool: bool = True,
        norm_cfg: dict = dict(type='SyncBN'),
        init_cfg: Optional[Union[dict, List[dict]]] = [
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]
    ) -> None:
        super().__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(in_channels, hid_channels)
        self.bn0 = build_norm_layer(
            dict(**norm_cfg, momentum=0.001, affine=False), hid_channels)[1]
        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function.

        Args:
            x (List[torch.Tensor]): The feature map of backbone.

        Returns:
            List[torch.Tensor]: The output features.
        """
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]
