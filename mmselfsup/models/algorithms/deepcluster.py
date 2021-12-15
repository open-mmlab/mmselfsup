# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from ..utils import Sobel
from .base import BaseModel


@ALGORITHMS.register_module()
class DeepCluster(BaseModel):
    """DeepCluster.

    Implementation of `Deep Clustering for Unsupervised Learning
    of Visual Features <https://arxiv.org/abs/1807.05520>`_.
    The clustering operation is in `core/hooks/deepcluster_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter on images.
            Defaults to True.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 with_sobel=True,
                 neck=None,
                 head=None,
                 init_cfg=None):
        super(DeepCluster, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

        # re-weight
        self.num_classes = self.head.num_classes
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32)
        self.loss_weight /= self.loss_weight.sum()

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

    def forward_train(self, img, pseudo_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            pseudo_label (Tensor): Label assignments.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        if self.with_neck:
            x = self.neck(x)
        outs = self.head(x)
        loss_inputs = (outs, pseudo_label)
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
        if self.with_neck:
            x = self.neck(x)
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def set_reweight(self, labels, reweight_pow=0.5):
        """Loss re-weighting.

        Re-weighting the loss according to the number of samples in each class.

        Args:
            labels (numpy.ndarray): Label assignments.
            reweight_pow (float): The power of re-weighting. Defaults to 0.5.
        """
        histogram = np.bincount(
            labels, minlength=self.num_classes).astype(np.float32)
        inv_histogram = (1. / (histogram + 1e-10))**reweight_pow
        weight = inv_histogram / inv_histogram.sum()
        self.loss_weight.copy_(torch.from_numpy(weight))
        self.head.criterion = nn.CrossEntropyLoss(weight=self.loss_weight)
