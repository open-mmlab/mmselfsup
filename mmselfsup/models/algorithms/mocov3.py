# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class MoCoV3(BaseModel):
    """MoCo v3.

    Implementation of `An Empirical Study of Training Self-Supervised Vision
    Transformers <https://arxiv.org/abs/2104.02057>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        base_momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.99.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 backbone,
                 neck,
                 head,
                 base_momentum=0.99,
                 init_cfg=None,
                 **kwargs):
        super(MoCoV3, self).__init__(init_cfg)
        assert neck is not None
        self.base_encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.momentum_encoder = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.backbone = self.base_encoder[0]
        self.neck = self.base_encoder[1]
        assert head is not None
        self.head = build_head(head)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self):
        """Initialize base_encoder with init_cfg defined in backbone."""
        super(MoCoV3, self).init_weights()

        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the momentum encoder."""
        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                1. - self.momentum)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images. Typically these should be mean centered
                and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images. Typically these should
                be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        view_1 = img[0].cuda(non_blocking=True)
        view_2 = img[1].cuda(non_blocking=True)

        # compute query features, [N, C] each
        q1 = self.base_encoder(view_1)[0]
        q2 = self.base_encoder(view_2)[0]

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # here we use hook to update momentum encoder, which is a little
            # bit different with the official version but it has negligible
            # influence on the results
            k1 = self.momentum_encoder(view_1)[0]
            k2 = self.momentum_encoder(view_2)[0]

        losses = self.head(q1, k2)['loss'] + self.head(q2, k1)['loss']
        return dict(loss=losses)
