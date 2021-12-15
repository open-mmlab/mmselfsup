# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16

from ..builder import ALGORITHMS, build_backbone, build_head
from .base import BaseModel


@ALGORITHMS.register_module()
class RotationPred(BaseModel):
    """Rotation prediction.

    Implementation of `Unsupervised Representation Learning
    by Predicting Image Rotations <https://arxiv.org/abs/1803.07728>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, head=None, init_cfg=None):
        super(RotationPred, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
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

    def forward_train(self, img, rot_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            rot_label (Tensor): Labels for the rotations.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        outs = self.head(x)
        loss_inputs = (outs, rot_label)
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
        x = self.extract_feat(img)  # tuple
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, rot_label=None, mode='train', **kwargs):
        """Forward function to select mode and modify the input image shape.

        Args:
            img (Tensor): Input images, the shape depends on mode.
                Typically these should be mean centered and std scaled.
        """
        if mode != 'extract' and img.dim() == 5:  # Nx4xCxHxW
            assert rot_label.dim() == 2  # Nx4
            img = img.view(
                img.size(0) * img.size(1), img.size(2), img.size(3),
                img.size(4))  # (4N)xCxHxW
            rot_label = torch.flatten(rot_label)  # (4N)
        if mode == 'train':
            return self.forward_train(img, rot_label, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.extract_feat(img)
        else:
            raise Exception(f'No such mode: {mode}')
