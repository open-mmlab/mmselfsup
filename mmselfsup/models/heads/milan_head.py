# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MILANPretrainHead(BaseModule):

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss = MODELS.build(loss)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred / pred.norm(dim=2, keepdim=True)
        target = target / target.norm(dim=2, keepdim=True)
        loss = self.loss(pred, target)
        return loss
