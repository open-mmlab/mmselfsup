# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ALGORITHMS, build_backbone, build_head
from ..utils import Sobel
from .base import BaseModel
from timm.data.mixup import Mixup


@ALGORITHMS.register_module()
class VitClassification(BaseModel):
    """Simple image classification for Vit.

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
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
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
        img, label = self.mix_up(img, label)
        x = self.extract_feat(img)
        losses = self.head(x, label)
        return losses

    def forward_test(self, img, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        out = self.extract_feat(img)
        key = 'last_layer'
        out_tensor = out.cpu()
        return {key: out_tensor}
