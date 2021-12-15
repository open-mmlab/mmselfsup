# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class BYOL(BaseModel):
    """BYOL.

    Implementation of `Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning <https://arxiv.org/abs/2006.07733>`_.
    The momentum adjustment is in `core/hooks/byol_hook.py`.

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
                 base_momentum=0.996,
                 init_cfg=None,
                 **kwargs):
        super(BYOL, self).__init__(init_cfg)
        assert neck is not None
        self.online_net = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.target_net = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
            param_tgt.requires_grad = False

        self.backbone = self.online_net[0]
        self.neck = self.online_net[1]
        assert head is not None
        self.head = build_head(head)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        img_v1 = img[0]
        img_v2 = img[1]
        # compute online features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        # compute target features
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0]
            proj_target_v2 = self.target_net(img_v2)[0]

        losses = 2. * (
            self.head(proj_online_v1, proj_target_v2)['loss'] +
            self.head(proj_online_v2, proj_target_v1)['loss'])
        return dict(loss=losses)
