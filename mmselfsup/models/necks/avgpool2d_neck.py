# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class AvgPool2dNeck(BaseModule):
    """The average pooling 2d neck."""

    def __init__(self, output_size: int = 1) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function."""
        assert len(x) == 1
        return [self.avgpool(x[0])]
