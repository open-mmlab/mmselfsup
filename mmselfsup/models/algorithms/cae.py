# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
from torchvision.transforms import Normalize

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class CAE(BaseModel):
    """CAE.

    Implementation of `Context Autoencoder for Self-Supervised Representation
    Learning <https://arxiv.org/abs/2202.03026>`_.

    Args:
        backbone (dict, optional): Config dict for module of backbone.
        neck (dict, optional): Config dict for module of deep features to
            compact feature vectors. Defaults to None.
        head (dict, optional): Config dict for module of loss functions.
            Defaults to None.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.0.
        init_cfg (dict, optional): the config to control the initialization.
    """

    def __init__(self,
                 backbone: dict = None,
                 neck: dict = None,
                 head: dict = None,
                 base_momentum: float = 0.0,
                 init_cfg: dict = None,
                 **kwargs) -> None:
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

    def init_weights(self) -> None:
        super().init_weights()
        self._init_teacher()

    def _init_teacher(self) -> None:
        # init the weights of teacher with those of backbone
        for param_backbone, param_teacher in zip(self.backbone.parameters(),
                                                 self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_backbone.data)
            param_teacher.requires_grad = False

    def momentum_update(self) -> None:
        """Momentum update of the teacher network."""
        for param_bacbone, param_teacher in zip(self.backbone.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + \
                param_bacbone.data * (1. - self.momentum)

    def extract_feat(self, img: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:

        x = self.backbone(img, mask)
        return x

    def forward_train(self, samples: Sequence, **kwargs) -> dict:
        img, img_target, mask = samples

        # normalize images and the images to get the target
        img_list = [self.img_norm(x).unsqueeze(0) for x in img]
        img = torch.cat(img_list)
        img_target = 0.8 * img_target + 0.1

        mask = mask.flatten(1).to(torch.bool)

        unmasked = self.backbone(img, mask)

        # get the latent prediction for the masked patches
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

        # input the unmasked tokens and masked tokens to the decoder
        logits, latent_pred = self.neck(unmasked[:, 1:], pos_embed_masked,
                                        pos_embed_unmasked)

        logits = logits.view(-1, logits.shape[-1])

        losses = self.head(img_target, logits, latent_pred, latent_target,
                           mask)
        return losses
