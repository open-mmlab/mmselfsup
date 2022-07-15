# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.data import SelfSupDataSample
from mmselfsup.registry import MODELS
from .base import BaseModel


@MODELS.register_module()
class SwAV(BaseModel):
    """SwAV.

    Implementation of `Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments <https://arxiv.org/abs/2006.09882>`_. The queue is
    built in `core/hooks/swav_hook.py`.
    """

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        x = self.backbone(batch_inputs[0])
        return x

    def loss(self, batch_inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """Forward computation during training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        assert isinstance(batch_inputs, list)
        # multi-res forward passes
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([input.shape[-1] for input in batch_inputs]),
                return_counts=True)[1], 0)
        start_idx = 0
        output = []
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(batch_inputs[start_idx:end_idx]))
            output.append(_out)
            start_idx = end_idx
        output = self.neck(output)[0]

        loss = self.head(output)
        losses = dict(loss=loss)
        return losses
