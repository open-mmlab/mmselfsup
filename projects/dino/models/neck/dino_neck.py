# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class DINONeck(BaseModule):

    def __init__(self) -> None:
        super().__init__()
        # TODO: implement the initialization function

    def forward(self):
        """Forward function of DINO Neck."""
        # TODO: implement the forward pass of neck here

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)