# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class GreenMIMHead(BaseModule):
    """Pretrain Head for SimMIM.

    Args:
        patch_size (int): Patch size of each token.
        loss (dict): The config for loss.
    """

    def __init__(self, patch_size, norm_pix_loss, loss: dict) -> None:
        super().__init__()
        self.loss = MODELS.build(loss)
        self.final_patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss


    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.norm_pix_loss:  # 这个是False
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2   # 用的MSE loss，这部分非常简单
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss
