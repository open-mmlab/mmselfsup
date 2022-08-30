# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import LabelData

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class RotationPred(BaseModel):
    """Rotation prediction.

    Implementation of `Unsupervised Representation Learning by Predicting Image
    Rotations <https://arxiv.org/abs/1803.07728>`_.
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

        x = self.backbone(inputs[0])

        rot_label = [
            data_sample.pseudo_label.rot_label for data_sample in data_samples
        ]
        rot_label = torch.flatten(torch.stack(rot_label, 0))  # (4N, )
        loss = self.head(x, rot_label)
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

        x = self.backbone(inputs[0])  # tuple
        outs = self.head.logits(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        outs = [torch.chunk(out, len(outs[0]) // 4, 0) for out in outs]

        for i in range(len(outs[0])):
            prediction_data = {key: out[i] for key, out in zip(keys, outs)}
            prediction = LabelData(**prediction_data)
            data_samples[i].pred_score = prediction
        return data_samples
