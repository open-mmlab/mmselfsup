# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule
from torch.nn import functional as F

from ..builder import HEADS


@HEADS.register_module()
class SimMIMHead(BaseModule):
    """Pretrain Head for SimMIM.

    Args:
        patch_size (int): Patch size of each token.
        encoder_in_channels (int): Number of input channels for encoder.
    """

    def __init__(self, patch_size: int, encoder_in_channels: int) -> None:
        super(SimMIMHead, self).__init__()
        self.patch_size = patch_size
        self.encoder_in_channels = encoder_in_channels

    def forward(self, x: torch.Tensor, x_rec: torch.Tensor,
                mask: torch.Tensor) -> dict:
        losses = dict()

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(
            self.patch_size, 2).unsqueeze(1).contiguous()
        loss_rec = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_rec * mask).sum() / (mask.sum() +
                                          1e-5) / self.encoder_in_channels

        losses['loss'] = loss

        return losses
