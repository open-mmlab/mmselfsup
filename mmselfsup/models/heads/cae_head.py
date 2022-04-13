# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import BaseModule
from torch.nn import functional as F
from torch import nn

from ..builder import HEADS
from ..utils import Encoder


@HEADS.register_module()
class CAEHead(BaseModule):
    """Pretrain Head for CAE.

    Args:
        tokenizer_path (str): The path of the tokenizer.
    """

    def __init__(self,
                 tokenizer_path: str,
                 lamb: float,
                 init_cfg: dict = None) -> None:
        super(CAEHead, self).__init__(init_cfg=init_cfg)
        self.tokenizer_path = tokenizer_path
        self.lamb = lamb
        self.encoder = self._load_encoder()
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss()

    def _load_encoder(self) -> nn.Module:
        encoder = Encoder()
        state_dict = torch.load(self.tokenizer_path)
        encoder.load_state_dict(state_dict)
        return encoder

    @torch.no_grad()
    def _generate_target(self, img_target):
        logits = self.encoder(img_target)
        target = torch.argmax(logits, dim=1)
        return target.flatten(1)

    def forward(self, img_target, outputs, latent_pred, latent_target,
                mask) -> dict:
        losses = dict()
        target = self._generate_target(img_target)
        target = target[mask]
        loss_main = self.loss_cross_entropy(outputs, target)
        loss_align = self.loss_mse(latent_pred,
                                   latent_target.detach()) * self.lamb

        losses['loss'] = loss_main + loss_align
        losses['main'] = loss_main
        losses['align'] = loss_align

        return losses
