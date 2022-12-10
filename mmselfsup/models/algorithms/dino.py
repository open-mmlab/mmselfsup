# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from ..utils import GatherLayer
from .base import BaseModel


@MODELS.register_module()
class DINO(BaseModel):


    @staticmethod
    def __init__(self,
                 backbone=None,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 momentum_teacher=0.996,
                 epoch=0,
                 drop_path_rate=0.1,
                 norm_last_layer=True):

        super(DINO, self).__init__()
        self.m = momentum_teacher
        self.epoch = epoch

        # create the teacher and student
        self.teacher = nn.Sequential(self.backbone(backbone),self.neck(neck))
        self.student = nn.Sequential(self.backbone(backbone),self.neck(neck))


    def _momentum_update_teacher(self):
        """
        Momentum update of the teacher
        """
        for param_q, param_k in zip(self.student.parameters(),self.teacher.parameters()):
            param_k=(param_k * self.m + param_q * (1. - self.m))

    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


    def loss(self, student_output, teacher_output, epoch)-> Dict[str, torch.Tensor]:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss