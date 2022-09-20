# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import LabelSmoothLoss
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule
from torch import nn

from ..builder import HEADS


@HEADS.register_module()
class MaskFeatPretrainHead(BaseModule):
    """Pre-training head for MaskFeat.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
            Defaults to 768.
        hog_dim (int): The dim of the hog feature. Defaults to 108.
    """

    def __init__(self, embed_dim: int = 768, hog_dim: int = 108) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, hog_dim)

    def init_weights(self) -> None:
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=0.02)

    def loss(self, pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> dict:
        """Compute the loss.

        Args:
            pred (torch.Tensor): Input prediction of shape (N, L, C).
            target (torch.Tensor): Input target of shape (N, L, C).
            mask (torch.Tensor): Input mask of shape (N, L, 1).
        Returns:
            dict[str, torch.Tensor]: A dictionary of loss components.
        """
        losses = dict()

        pred = pred[mask]
        target = target[mask]
        loss = ((pred - target)**2).mean(-1).mean()

        losses['loss'] = loss
        return losses

    def forward(self, latent: torch.Tensor, hog: torch.Tensor,
                mask: torch.Tensor) -> dict:
        """Pre-training head for MaskFeat.

        Args:
            latent (torch.Tensor): Input latent of shape (N, 1+L, C).
            hog (torch.Tensor): Input hog feature of shape (N, L, C).
            mask (torch.Tensor): Input mask of shape (N, H, W).
        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        latent = self.head(latent)
        mask = mask.flatten(1).bool()
        losses = self.loss(latent[:, 1:], hog, mask)

        return losses


@HEADS.register_module()
class MaskFeatFinetuneHead(BaseModule):
    """Fine-tuning head for MaskFeat.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
        label_smooth_val (float): The degree of label smoothing.
            Defaults to 0.1.
    """

    def __init__(self,
                 embed_dim: int,
                 num_classes: int = 1000,
                 label_smooth_val: float = 0.1) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes, bias=True)
        self.act = nn.Softmax(dim=1)
        self.criterion = LabelSmoothLoss(label_smooth_val, num_classes)

    def init_weights(self) -> None:
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=.02)

    def forward(self, x: torch.Tensor) -> list:
        """"Get the logits."""
        outputs = self.head(x)
        if not self.training:
            outputs = self.act(outputs)
        return [outputs]

    def loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> dict:
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs[0], labels)

        return losses
