# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import HEADS


@HEADS.register_module()
class PixelHead(BaseModule):
    """Head for pixel_level reconstruction.

    The MSE loss is implemented in this head and is used in generative methods, e.g. MAE
    """

    def __init__(self):
        super(PixelHead, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, outputs, labels):

        losses = dict()

        losses['loss'] = self.criterion(outputs, labels)

        return losses
