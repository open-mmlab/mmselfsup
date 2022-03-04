# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class SimMIMNeck(BaseModule):

    def __init__(self, in_channels, encoder_stride):
        super(SimMIMNeck, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=encoder_stride**2 * 3,
                kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, x):

        x = self.decoder(x)

        return x
