# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class MaskFeat(BaseModel):
    """MaskFeat.

    Implementation of `Masked Feature Prediction for
    Self-Supervised Visual Pre-Training <https://arxiv.org/abs/2112.09133>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        head (dict):Config dict for module of head functions.
        hog_para (dict): Config dict for hog layer.
            dict['nbins', int]: Number of bin. Defaults to 9.
            dict['pool', float]: Number of cell. Defaults to 8.
            dict['gaussian_window', int]: Size of gaussian kernel.
                Defaults to 16.
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

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features from neck.

        Args:
            inputs (List[torch.Tensor]): The input images and mask.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.
        Returns:
            Tuple[torch.Tensor]: Neck outputs.
        """
        img = inputs[0]
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        latent = self.backbone(img, mask)
        return latent

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
        img = inputs[0]
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])

        latent = self.backbone(img, mask)
        hog = self.target_generator(img)

        loss = self.head(latent, hog, mask)
        losses = dict(loss=loss)
        return losses
