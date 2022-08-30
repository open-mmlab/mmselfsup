# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class CAE(BaseModel):
    """CAE.

    Implementation of `Context Autoencoder for Self-Supervised Representation
    Learning <https://arxiv.org/abs/2202.03026>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of neck.
        head (dict): Config dict for module of head functions.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.0.
        data_preprocessor (dict, optional): The config for preprocessing
            input data. If None or no specified type, it will use
            "SelfSupDataPreprocessor" as type.
            See :class:`SelfSupDataPreprocessor` for more details.
            Defaults to None.
        init_cfg (Union[List[dict], dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.0,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.momentum = base_momentum
        self.teacher = MODELS.build(backbone)

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()
        self._init_teacher()

    def _init_teacher(self) -> None:
        """Init the weights of teacher with those of backbone."""
        for param_backbone, param_teacher in zip(self.backbone.parameters(),
                                                 self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_backbone.data)
            param_teacher.requires_grad = False

    def momentum_update(self) -> None:
        """Momentum update of the teacher network."""
        for param_bacbone, param_teacher in zip(self.backbone.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + \
                param_bacbone.data * (1. - self.momentum)

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
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])

        mask = mask.flatten(1).to(torch.bool)

        unmasked = self.backbone(inputs[0], mask)

        # get the latent prediction for the masked patches
        with torch.no_grad():
            # inputs[0] is the prediction image
            latent_target = self.teacher(inputs[0], ~mask)
            latent_target = latent_target[:, 1:, :]
            self.momentum_update()

        pos_embed = self.backbone.pos_embed.expand(inputs[0].shape[0], -1, -1)
        pos_embed_masked = pos_embed[:,
                                     1:][mask].reshape(inputs[0].shape[0], -1,
                                                       pos_embed.shape[-1])
        pos_embed_unmasked = pos_embed[:, 1:][~mask].reshape(
            inputs[0].shape[0], -1, pos_embed.shape[-1])

        # input the unmasked tokens and masked tokens to the decoder
        logits, latent_pred = self.neck(unmasked[:, 1:], pos_embed_masked,
                                        pos_embed_unmasked)

        logits = logits.view(-1, logits.shape[-1])
        # inputs[1] is the target image
        loss_main, loss_align = self.head(logits, inputs[1], latent_pred,
                                          latent_target, mask)
        losses = dict()

        losses['loss'] = loss_main + loss_align
        losses['main'] = loss_main
        losses['align'] = loss_align
        return losses
