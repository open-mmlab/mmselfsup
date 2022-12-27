# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class VideoMAEPretrainHead(BaseModule):

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 patch_size: int = 16) -> None:
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.loss = MODELS.build(loss)

    def patchify(self, video: torch.Tensor) -> torch.Tensor:
        """

        Args:
            video (torch.Tensor): A batch of videos,
                of shape B x T x C x H x W.

        Returns:
            torch.Tensor: Patchified videos. The shape is B x T x L x D.
        """
        p = self.patch_size
        assert video.shape[3] == video.shape[4] and video.shape[3] % p == 0
        h = w = video.shape[3] // p
        x = video.reshape(
            shape=(video.shape[0], video.shape[1], 3, h, p, w, p))
        x = torch.einsum('ntchpwq->nthwpqc', x)
        x = x.reshape(shape=(video.shape[0], video.shape[1], h * w, p**2 * 3))
        return x

    def contruct_target(self, target: torch.Tensor) -> torch.Tensor:
        target = self.patchify(target)
        if self.norm_pix:
            # normalize the target video
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return target

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        target = self.construct_target(target)
        loss = self.loss(pred, target, mask)

        return loss
