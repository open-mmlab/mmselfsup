# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.data import LabelData

from mmselfsup.core import SelfSupDataSample
from mmselfsup.registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class ODC(BaseModel):
    """ODC.

    Official implementation of `Online Deep Clustering for Unsupervised
    Representation Learning <https://arxiv.org/abs/2006.10645>`_.
    The operation w.r.t. memory bank and loss re-weighting is in
     `core/hooks/odc_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors.
        head (dict): Config dict for module of head functions.
        memory_bank (dict): Module of memory banks. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict or nn.Module, optional): Config to preprocess
            images. Defaults to None.
        init_cfg (dict or List[dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 memory_bank: dict,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        # build memory
        self.memory_bank = MODELS.build(memory_bank)

        # set re-weight tensors
        self.num_classes = self.head.num_classes
        self.loss_weight = torch.ones((self.num_classes, ),
                                      dtype=torch.float32).cuda()
        self.loss_weight /= self.loss_weight.sum()

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        x = self.backbone(batch_inputs[0])
        return x

    def loss(self, batch_inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        feature = self.extract_feat(batch_inputs)
        idx = [data_sample.sample_idx.value for data_sample in data_samples]
        idx = torch.cat(idx)
        if self.with_neck:
            feature = self.neck(feature)
        if self.memory_bank.label_bank.is_cuda:
            loss_inputs = (feature, self.memory_bank.label_bank[idx])
        else:
            loss_inputs = (feature,
                           self.memory_bank.label_bank[idx.cpu()].cuda())

        loss = self.head(*loss_inputs)
        losses = dict(loss=loss)

        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature[0].detach())
        losses['change_ratio'] = change_ratio

        return losses

    def predict(self, batch_inputs: List[torch.Tensor],
                data_samples: List[SelfSupDataSample],
                **kwargs) -> List[SelfSupDataSample]:
        """The forward function in testing.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            List[SelfSupDataSample]: The prediction from model.
        """
        feature = self.extract_feat(batch_inputs)  # tuple
        if self.with_neck:
            feature = self.neck(feature)
        outs = self.head.logits(feature)
        keys = [f'head{i}' for i in self.backbone.out_indices]

        for i in range(len(outs)):
            prediction_data = {key: out for key, out in zip(keys, outs)}
            prediction = LabelData(**prediction_data)
            data_samples[i].pred_label = prediction
        return data_samples
