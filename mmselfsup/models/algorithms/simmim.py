# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class SimMIM(BaseModel):
    """SimMIM.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self, backbone=None, neck=None, head=None, init_cfg=None):
        super(SimMIM, self).__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        return self.backbone(img)

    def forward_train(self, x, **kwargs):
        img, mask = x

        img_latent = self.backbone(img, mask)
        img_rec = self.neck(img_latent[0])
        losses = self.head(img, img_rec, mask)

        return losses
