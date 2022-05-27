# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.data import InstanceData

from mmselfsup.core import SelfSupDataSample
from ..builder import (ALGORITHMS, build_backbone, build_head, build_loss,
                       build_neck)
from ..utils import Sobel
from .base import BaseModel


@ALGORITHMS.register_module()
class DeepCluster(BaseModel):
    """DeepCluster.

    Implementation of `Deep Clustering for Unsupervised Learning
    of Visual Features <https://arxiv.org/abs/1807.05520>`_.
    The clustering operation is in `core/hooks/deepcluster_hook.py`.

    Args:
        backbone (Dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter on images.
            Defaults to True.
        neck (Dict, optional): Config dict for module of deep features to
            compact feature vectors. Defaults to None.
        head (Dict, optional): Config dict for module of loss functions.
        loss (Dict, optional): Config dict for module of loss functions.
            Defaults to None.
        preprocess_cfg (Dict): Config to preprocess images.
        init_cfg (Union[List[Dict], Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Dict,
                 with_sobel: Optional[bool] = True,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 loss: Optional[Dict] = None,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[List[Dict], Dict]] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        assert loss is not None
        self.loss = build_loss(loss)

        # re-weight
        self.num_classes = self.head.num_classes
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32)
        self.loss_weight /= self.loss_weight.sum()

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        if self.with_sobel:
            img = self.sobel_layer(inputs[0])
        x = self.backbone(img)
        return x

    def forward_train(self, inputs: List[torch.Tensor],
                      data_samples: List[SelfSupDataSample],
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Forward computation during training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        pseudo_label = torch.cat(
            [data_sample.pseudo_label.label for data_sample in data_samples])
        x = self.extract_feat(inputs, data_samples)
        if self.with_neck:
            x = self.neck(x)
        outs = self.head(x)
        self.loss.class_weight = self.loss_weight
        loss = self.loss(outs[0], pseudo_label)
        losses = dict(loss=loss)
        return losses

    def forward_test(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwargs) -> List[SelfSupDataSample]:
        """The forward function in testing
        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.
        Returns:
            List[SelfSupDataSample]: The prediction from model.
        """
        x = self.extract_feat(inputs, data_samples)  # tuple
        if self.with_neck:
            x = self.neck(x)
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]

        for i in range(x[0].shape[0]):
            prediction_data = {key: out[i] for key, out in zip(keys, outs)}
            prediction = InstanceData(**prediction_data)
            data_samples[i].prediction = prediction
        return data_samples
