# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torchvision.transforms import Normalize

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class CAE(BaseModel):
    """CAE.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.996.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 base_momentum=0.0,
                 init_cfg=None,
                 **kwargs):
        super(CAE, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        self.teacher = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

        self.momentum = base_momentum

        self.img_norm = Normalize(
            mean=torch.tensor((0.485, 0.456, 0.406)),
            std=torch.tensor((0.229, 0.224, 0.225)))

    def init_weights(self):
        super().init_weights()
        self._init_teacher()

    def _init_teacher(self):
        for param_backbone, param_teacher in zip(self.backbone.parameters(),
                                                 self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_backbone.data)
            param_teacher.requires_grad = False

    def momentum_update(self):
        """Momentum update of the target network."""
        for param_bacbone, param_teacher in zip(self.backbone.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + \
                param_bacbone.data * (1. - self.momentum)

    def extract_feat(self, img, mask):

        x = self.backbone(img, mask)
        return x

    def forward_train(self, img, **kwargs):
        img, img_target, mask = img
        img = self.img_norm(img)
        img_target = 0.8 * img_target + 0.1

        mask = mask.flatten(1).to(torch.bool)

        unmasked = self.backbone(img, mask)
        with torch.no_grad():
            latent_target = self.teacher(img, ~mask)
            latent_target = latent_target[:, 1:, :]
            self.momentum_update()

        pos_embed = self.backbone.pos_embed.expand(img.shape[0], -1, -1)
        pos_embed_masked = pos_embed[:,
                                     1:][mask].reshape(img.shape[0], -1,
                                                       pos_embed.shape[-1])
        pos_embed_unmasked = pos_embed[:, 1:][~mask].reshape(
            img.shape[0], -1, pos_embed.shape[-1])

        logits, latent_pred = self.neck(unmasked[:, 1:], pos_embed_masked,
                                        pos_embed_unmasked)

        logits = logits.view(-1, logits.shape[-1])

        losses = self.head(img_target, logits, latent_pred, latent_target,
                           mask)
        return losses
