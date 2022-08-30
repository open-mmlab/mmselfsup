# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import LabelData

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class RelativeLoc(BaseModel):
    """Relative patch location.

    Implementation of `Unsupervised Visual Representation Learning by Context
    Prediction <https://arxiv.org/abs/1505.05192>`_.
    """

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.

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
        x1 = self.backbone(inputs[0])
        x2 = self.backbone(inputs[1])
        x = (torch.cat((x1[0], x2[0]), dim=1), )
        x = self.neck(x)
        patch_label = [
            data_sample.pseudo_label.patch_label
            for data_sample in data_samples
        ]

        patch_label = torch.flatten(torch.stack(patch_label, 0))
        loss = self.head(x, patch_label)
        losses = dict(loss=loss)
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
        x1 = self.backbone(inputs[0])
        x2 = self.backbone(inputs[1])
        x = (torch.cat((x1[0], x2[0]), dim=1), )
        x = self.neck(x)
        outs = self.head.logits(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        outs = [torch.chunk(out, len(outs[0]) // 8, 0) for out in outs]

        for i in range(len(outs[0])):
            prediction_data = {key: out[i] for key, out in zip(keys, outs)}
            prediction = LabelData(**prediction_data)
            data_samples[i].pred_label = prediction
        return data_samples
