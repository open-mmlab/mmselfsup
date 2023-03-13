# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models import BaseModel, CosineEMA


@MODELS.register_module()
class DINO(BaseModel):

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 pretrained: Optional[str] = None,
                 base_momentum: float = 0.99,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.teacher = CosineEMA(
            nn.Sequential(self.backbone, self.neck), momentum=base_momentum)
        self.fix_teacher()

    def fix_teacher(self) -> None:
        """Fix the teacher model."""
        for param in self.teacher.parameters():
            param.requires_grad = False

    def loss(self, inputs: torch.Tensor,
             data_samples: List[SelfSupDataSample]) -> dict:

        global_crops = torch.cat(inputs[:2])
        local_crops = torch.cat(inputs[2:])

        # teacher forward
        teacher_output = self.teacher(global_crops)

        # student forward global
        student_output_global = self.backbone(global_crops)
        student_output_global = self.neck(student_output_global)

        # student forward local
        student_output_local = self.backbone(local_crops)
        student_output_local = self.neck(student_output_local)

        student_ouput = torch.cat(
            (student_output_global, student_output_local))

        # compute loss
        loss = self.head(student_ouput, teacher_output)

        return dict(loss=loss)
