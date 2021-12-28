# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import HEADS


@HEADS.register_module()
class MSEHead(BaseModule):
    """Head for pixel_level reconstruction.

    The MSE loss is implemented in this head and is used in generative methods,
    e.g. MAE
    """

    def __init__(self, patch_size=16, num_classes=768, embed_dim=384):
        super(MSEHead, self).__init__()
        self.criterion = nn.MSELoss()
        assert num_classes == 3 * patch_size**2
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, labels):

        losses = dict()
        outputs = self.head(x)
        losses['loss'] = self.criterion(outputs, labels)

        return losses
