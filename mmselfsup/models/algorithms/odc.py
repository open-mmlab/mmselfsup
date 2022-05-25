# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.data import InstanceData

from mmselfsup.core import SelfSupDataSample
from ..builder import (ALGORITHMS, build_backbone, build_head, build_loss,
                       build_memory, build_neck)
from ..utils import Sobel
from .base import BaseModel


@ALGORITHMS.register_module()
class ODC(BaseModel):
    """ODC.

    Official implementation of `Online Deep Clustering for Unsupervised
    Representation Learning <https://arxiv.org/abs/2006.10645>`_.
    The operation w.r.t. memory bank and loss re-weighting is in
     `core/hooks/odc_hook.py`.

    Args:
        backbone (Dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter on images.
            Defaults to False.
        neck (Dict, optional): Config dict for module of deep features to
            compact feature vectors. Defaults to None.
        head (Dict, optional): Config dict for module of head functions.
            Defaults to None.
        loss (Dict, optional): Config dict for module of loss functions.
        memory_bank (Dict, optional): Module of memory banks. Defaults to None.
        preprocess_cfg (Dict, optional): Config to preprocess images.
            Defaults to None.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Dict,
                 with_sobel: Optional[bool] = False,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 loss: Optional[Dict] = None,
                 memory_bank: Optional[Dict] = None,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
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
        assert memory_bank is not None
        self.memory_bank = build_memory(memory_bank)

        # set re-weight tensors
        self.num_classes = self.head.num_classes
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32).cuda()
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
        # forward & backward
        feature = self.extract_feat(inputs[0])
        idx = [data_sample.idx for data_sample in data_samples]
        idx = torch.cat(idx)
        if self.with_neck:
            feature = self.neck(feature)
        outs = self.head(feature)
        if self.memory_bank.label_bank.is_cuda:
            loss_inputs = (outs, self.memory_bank.label_bank[idx])
        else:
            loss_inputs = (outs, self.memory_bank.label_bank[idx.cpu()].cuda())
        self.loss.class_weight = self.loss_weight
        loss = self.loss(loss_inputs[0][0], loss_inputs[1])
        losses = dict(loss=loss)

        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature[0].detach())
        losses['change_ratio'] = change_ratio

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
        feature = self.extract_feat(inputs[0])  # tuple
        if self.with_neck:
            feature = self.neck(feature)
        outs = self.head(feature)
        keys = [f'head{i}' for i in range(len(outs))]

        for i in range(outs[0].shape[0]):
            prediction_data = {key: out[i] for key, out in zip(keys, outs)}
            prediction = InstanceData(**prediction_data)
            data_samples[i].prediction = prediction
        return data_samples
