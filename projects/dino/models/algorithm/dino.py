# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
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
        self.teacher=nn.Sequential(self.backbone(),self.neck())
        self.student=nn.Sequential(self.backbone())

    def momentum_update(self):
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.student.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        # TODO: implement the extract_feat function
        x=self.backbone(inputs[0])
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
        # TODO: implement the forward pass here

        assert isinstance(inputs, list)

        x1 = self.backbone(x1)
        x2 = self.backbone(self.neck(x2))

        loss=self.head(student_out,teacher_out)
        losses = dict(loss=loss)
        return losses

