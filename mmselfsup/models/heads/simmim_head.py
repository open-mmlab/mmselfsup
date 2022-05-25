# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule

from ..builder import HEADS


@HEADS.register_module()
class SimMIMHead(BaseModule):
    """Pretrain Head for SimMIM.

    Args:
        patch_size (int): Patch size of each token.
    """

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, mask: torch.Tensor) -> torch.Tensor:

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(
            self.patch_size, 2).unsqueeze(1).contiguous()

        return mask
