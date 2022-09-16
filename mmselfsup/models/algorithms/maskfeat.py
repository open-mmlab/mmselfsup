# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from ..utils.hog_layer import HOGLayerC
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
                 head: dict,
                 hog_para: dict,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.hog_layer = HOGLayerC(**hog_para)

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
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        img = inputs[0]
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])

        latent = self.backbone(img, mask)
        hog = self.hog_layer(img)
        loss = self.head(latent, hog, mask)
        losses = dict(loss=loss)
        return losses
