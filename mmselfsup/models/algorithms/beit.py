# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch

from mmselfsup.models.utils import RelativePositionBias
from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class BEiT(BaseModel):
    """BEiT.

    Implementation of `BEiT: BERT Pre-Training of Image Transformers
     <https://arxiv.org/abs/2106.08254>`_.

    Args:
        backbone (dict, optional): Config dict for module of backbone.
        neck (dict, optional): Config dict for module of deep features to
            compact feature vectors. Defaults to None.
        head (dict, optional): Config dict for module of loss functions.
            Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: dict = None,
                 use_shared_rel_pos_bias: bool = True,
                 **kwargs) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        if use_shared_rel_pos_bias:
            self.shared_rel_pos_bias = RelativePositionBias(
                window_size=self.backbone.patch_resolution,
                num_heads=self.backbone.arch_settings['num_heads'])
        else:
            self.shared_rel_pos_bias = None

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()

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
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        rel_pos_bias = self.shared_rel_pos_bias().to(mask.device)

        img_latent = self.backbone(batch_inputs[0], mask, rel_pos_bias)
        logits = self.neck(img_latent, rel_pos_bias=rel_pos_bias)

        # batch_inputs[1] is the target image
        loss = self.head(logits, batch_inputs[1], mask)
        if isinstance(loss, torch.Tensor):
            losses = dict(loss=loss)
            return losses
        elif isinstance(loss, Tuple):
            loss_1, loss_2 = loss[0], loss[1]
            losses = dict()
            losses['loss'] = loss_1 + loss_2
            losses['loss_1'] = loss_1
            losses['loss_2'] = loss_2
            return losses
