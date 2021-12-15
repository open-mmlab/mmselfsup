# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class RelativeLoc(BaseModel):
    """Relative patch location.

    Implementation of `Unsupervised Visual Representation Learning
    by Context Prediction <https://arxiv.org/abs/1505.05192>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super(RelativeLoc, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

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

    def forward_train(self, img, patch_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            patch_label (Tensor): Labels for the relative patch locations.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        img1, img2 = torch.chunk(img, 2, dim=1)
        x1 = self.extract_feat(img1)  # tuple
        x2 = self.extract_feat(img2)  # tuple
        x = (torch.cat((x1[0], x2[0]), dim=1), )
        x = self.neck(x)
        outs = self.head(x)
        loss_inputs = (outs, patch_label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        img1, img2 = torch.chunk(img, 2, dim=1)
        x1 = self.extract_feat(img1)  # tuple
        x2 = self.extract_feat(img2)  # tuple
        x = (torch.cat((x1[0], x2[0]), dim=1), )
        x = self.neck(x)
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]
        return dict(zip(keys, out_tensors))

    def forward(self, img, patch_label=None, mode='train', **kwargs):
        """Forward function to select mode and modify the input image shape.

        Args:
            img (Tensor): Input images, the shape depends on mode.
                Typically these should be mean centered and std scaled.
        """
        if mode != 'extract' and img.dim() == 5:  # Nx8x(2C)xHxW
            assert patch_label.dim() == 2  # Nx8
            img = img.view(
                img.size(0) * img.size(1), img.size(2), img.size(3),
                img.size(4))  # (8N)x(2C)xHxW
            patch_label = torch.flatten(patch_label)  # (8N)
        if mode == 'train':
            return self.forward_train(img, patch_label, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.extract_feat(img)
        else:
            raise Exception(f'No such mode: {mode}')
