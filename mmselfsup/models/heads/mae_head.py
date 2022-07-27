# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import LabelSmoothLoss
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule
from torch import nn

from ..builder import HEADS


@HEADS.register_module()
class MAEPretrainHead(BaseModule):
    """Pre-training head for MAE.

    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self, norm_pix: bool = False, patch_size: int = 16) -> None:
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs (torch.Tensor): The shape is (N, 3, H, W)
        Returns:
            x (torch.Tensor): The shape is (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The shape is (N, L, patch_size**2 *3)
        Returns:
            imgs (torch.Tensor): The shape is (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, pred: torch.Tensor,
                mask: torch.Tensor) -> dict:
        losses = dict()
        target = self.patchify(x)
        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        losses['loss'] = loss
        return losses


@HEADS.register_module()
class MAEFinetuneHead(BaseModule):
    """Fine-tuning head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000, label_smooth_val=0.1):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.criterion = LabelSmoothLoss(label_smooth_val, num_classes)

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, x):
        """"Get the logits."""
        outputs = self.head(x)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs[0], labels)

        return losses


@HEADS.register_module()
class MAELinprobeHead(BaseModule):
    """Linear probing head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.bn = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=0.01)

    def forward(self, x):
        """"Get the logits."""
        x = self.bn(x)
        outputs = self.head(x)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs[0], labels)

        return losses
