# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F
from mmcv.runner import BaseModule

from ..builder import HEADS


@HEADS.register_module()
class SimMIMPretrainHead(BaseModule):

    def __init__(self, patch_size, encoder_in_channels):
        super(SimMIMPretrainHead, self).__init__()
        self.patch_size = patch_size
        self.encoder_in_channels = encoder_in_channels

    def patchify(self, x, mask: torch.Tensor):

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(
            self.patch_size, 2).unsqueeze(1).contiguous()
        return x

    def forward(self, x, x_rec, mask):
        losses = dict()

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(
            self.patch_size, 2).unsqueeze(1).contiguous()
        loss_rec = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_rec * mask).sum() / (mask.sum() +
                                          1e-5) / self.encoder_in_channels

        losses['loss'] = loss

        return losses
