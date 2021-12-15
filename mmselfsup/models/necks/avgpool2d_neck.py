# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class AvgPool2dNeck(BaseModule):
    """The average pooling 2d neck."""

    def __init__(self, output_size=1):
        super(AvgPool2dNeck, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        """Forward function."""
        assert len(x) == 1
        return [self.avgpool(x[0])]
