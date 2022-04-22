# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models.utils import Augments

from ..builder import ALGORITHMS, build_backbone, build_head
from ..utils import Sobel
from .base import BaseModel


@ALGORITHMS.register_module()
class Classification(BaseModel):
    """Simple image classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 with_sobel=False,
                 head=None,
                 train_cfg=None,
                 init_cfg=None):
        super(Classification, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            self.augments = Augments(augments_cfg)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        if self.with_sobel:
            img = self.sobel_layer(img)
        x = self.backbone(img)
        return x

    def forward_train(self, img, label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if self.augments is not None:
            img, label = self.augments(img, label)
        x = self.extract_feat(img)
        outs = self.head(x)
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        x = self.extract_feat(img)  # tuple
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))
