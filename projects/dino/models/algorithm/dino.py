# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models import BaseModel


@MODELS.register_module()
class DINO(BaseModel):

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
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

        # TODO: implement the initialization function

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        # TODO: implement the extract_feat function

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
        # TODO: implement the forward pass here
