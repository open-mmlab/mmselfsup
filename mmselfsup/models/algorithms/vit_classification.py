# Copyright (c) OpenMMLab. All rights reserved.
from ..utils import Mixup

from ..builder import ALGORITHMS, build_backbone, build_head
from .base import BaseModel


@ALGORITHMS.register_module()
class VitClassification(BaseModel):
    """Simple image classification for ViT.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 head=None,
                 init_cfg=None,
                 mixup_alpha=None,
                 cutmix_alpha=None,
                 cutmix_minmax=None,
                 prob=None,
                 switch_prob=None,
                 mode=None,
                 label_smoothing=None,
                 num_classes=None):
        super(VitClassification, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)
        self.mix_up = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=cutmix_minmax,
            prob=prob,
            switch_prob=switch_prob,
            mode=mode,
            label_smoothing=label_smoothing,
            num_classes=num_classes)

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
        img, label = self.mix_up(img, label)
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
        x = self.extract_feat(img)
        outs = self.head(x)
        key = 'last_layer'
        out_tensor = outs.cpu()
        return {key: out_tensor}
