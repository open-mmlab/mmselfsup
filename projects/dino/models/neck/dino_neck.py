# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS
import torch
from torch import nn


@MODELS.register_module()
class DINONeck(BaseModule):

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, bottleneck_channels: int) -> None:
        self.mlp = nn.Sequential(*[
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, bottleneck_channels),
        ])

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_channels, out_channels, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
