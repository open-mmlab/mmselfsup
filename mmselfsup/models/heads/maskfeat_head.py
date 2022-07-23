# Copyright (c) OpenMMLab. All rights reserved.
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

    def __init__(self, embed_dim=768, hog_dim=108):
        super().__init__()
        self.head = nn.Linear(embed_dim, hog_dim)

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=2e-5)

    def loss(self, pred, target, mask):
        """Compute the loss.

        Args:
            pred (torch.Tensor): Input prediction of shape (N, C, H, W).
            target (torch.Tensor): Input target of shape (N, C, H, W).
            mask (torch.Tensor): Input mask of shape (N, H, W).

        Returns:
            dict[str, torch.Tensor]: A dictionary of loss components.
        """
        losses = dict()

        if pred.is_cuda:
            target = target.cuda(pred.device)

        loss = (pred - target)**2
        loss = loss.mean(dim=1)

        loss = (loss * mask).sum() / (mask.sum() + 1e-5)
        losses['loss'] = loss
        return losses

    def forward(self, latent, hog, mask):
        """Pre-training head for MaskFeat.

        Args:
            latent (torch.Tensor): Input latent of shape (N, C, H, W).
            hog (torch.Tensor): Input hog feature of shape (N, C, H, W).
            mask (torch.Tensor): Input mask of shape (N, H, W).
        Returns:
            dict[str, torch.Tensor]: A dictionary of loss components.
        """
        N, C, H, W = hog.shape
        latent = self.head(latent)
        latent = latent.permute(0, 2, 1).reshape(N, C, H, W)
        mask = mask.reshape(N, H, W)
        losses = self.loss(latent, hog, mask)
        return losses


@HEADS.register_module()
class MaskFeatFinetuneHead(BaseModule):
    """Fine-tuning head for MaskFeat.

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
