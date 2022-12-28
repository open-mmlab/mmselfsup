# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class VideoMAEPretrainHead(BaseModule):

    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 tubelet_size: int = 2,
                 patch_size: int = 16) -> None:
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.loss = MODELS.build(loss)

    def patchify(self, video: torch.Tensor) -> torch.Tensor:
        """

        Args:
            video (torch.Tensor): A batch of videos, of shape
            B x C x T x H x W. C is the channel, T is the temporal length

        Returns:
            torch.Tensor: Patchified videos. The shape is B x T x L x D.
        """
        p = self.patch_size
        ts = self.tubelet_size
        B, C, T, H, W = video.shape
        assert H == W and H % p == 0 and C == 3
        # number of patches in height and width
        h = w = H // p
        # number of tubelet in temporal dimension
        t = T // ts

        # video shape (B, 3, T, H, W)
        x = video.reshape(shape=(B, 3, t, ts, h, p, w, p))
        # 'b c ts hp wq->b t hw spq c'
        x = torch.einsum('bctshpwq->bthwspqc', x)
        # (B, num_token, num_pixel_per_token, 3)
        x = x.reshape(shape=(B, t * h * w, ts * p * p, 3))
        return x

    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        target = self.patchify(target)
        if self.norm_pix:
            # normalize the target video, different from the mae
            mean = target.mean(dim=-2, keepdim=True)
            var = target.var(dim=-2, unbiased=True, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        B, T, L, C = target.shape
        target = target.view(B, T, L * C)
        return target

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        target = self.construct_target(target)
        B, _, C = target.shape
        target = target[mask].reshape(B, -1, C)
        loss = self.loss(pred, target)

        return loss
