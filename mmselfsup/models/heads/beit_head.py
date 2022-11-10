# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class BEiTHead(BaseModule):
    """Pretrain Head for BEiT.

    Compute the cross entropy loss. In addition, this head also
    generates the prediction target generated by dalle.

    Args:
        loss (dict): The config of loss.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_embed: int,
                 loss: dict,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.cls_head = nn.Linear(embed_dims, num_embed)
        self.loss = MODELS.build(loss)

    def forward(self,
                feats: torch.Tensor,
                feats_cls_pt: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor,
                return_all_tokens=False) -> torch.Tensor:
        """Generate loss.

        Args:
            logits (torch.Tensor): Logits generated by decoder.
            target (torch.Tensor): Target generated by target_generator.
        """
        # shared cls head
        logits, logits_cls_pt = self.cls_head(feats), self.cls_head(
            feats_cls_pt)
        logits = logits.view(-1, logits.shape[-1])
        logits_cls_pt = logits_cls_pt.view(-1, logits_cls_pt.shape[-1])

        if not return_all_tokens:
            mask = mask.flatten(0).to(torch.bool)
            target = target.view(-1, 1)
            target = target[mask]
            logits = (logits[mask], logits_cls_pt[mask])

        loss = self.loss(logits, target.squeeze(-1))
        return loss
