# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.device import get_device
from mmengine.structures import LabelData

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class ODC(BaseModel):
    """ODC.

    Official implementation of `Online Deep Clustering for Unsupervised
    Representation Learning <https://arxiv.org/abs/2006.10645>`_.
    The operation w.r.t. memory bank and loss re-weighting is in
    `engine/hooks/odc_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors.
        head (dict): Config dict for module of head functions.
        memory_bank (dict): Config dict for module of memory bank.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 memory_bank: dict,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
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
        self.register_buffer(
            'loss_weight', torch.ones((self.num_classes, ),
                                      dtype=torch.float32))
        self.loss_weight /= self.loss_weight.sum()

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        x = self.backbone(inputs[0])
        return x

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        feature = self.extract_feat(inputs)
        idx = [data_sample.sample_idx.value for data_sample in data_samples]
        idx = torch.cat(idx)

        if self.with_neck:
            feature = self.neck(feature)

        loss_inputs = (feature,
                       self.memory_bank.label_bank[idx].to(get_device()))
        loss = self.head(*loss_inputs)
        losses = dict(loss=loss)

        # update samples memory
        change_ratio = self.memory_bank.update_samples_memory(
            idx, feature[0].detach())
        losses['change_ratio'] = change_ratio

        return losses

    def predict(self, inputs: List[torch.Tensor],
                data_samples: List[SelfSupDataSample],
                **kwargs) -> List[SelfSupDataSample]:
        """The forward function in testing.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            List[SelfSupDataSample]: The prediction from model.
        """
        feature = self.extract_feat(inputs)  # tuple
        if self.with_neck:
            feature = self.neck(feature)
        outs = self.head.logits(feature)
        keys = [f'head{i}' for i in self.backbone.out_indices]

        for i in range(len(outs)):
            prediction_data = {key: out for key, out in zip(keys, outs)}
            prediction = LabelData(**prediction_data)
            data_samples[i].pred_label = prediction
        return data_samples
