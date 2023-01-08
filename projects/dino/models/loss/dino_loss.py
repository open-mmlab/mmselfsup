# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class DINOLoss(BaseModule):

    def __init__(self) -> None:
        super().__init__()
        # TODO: implement the initialization function

    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self):
        """Forward function of DINO Loss."""
        # TODO: implement the forward pass of loss here
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
