# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch

from mmselfsup.core import SelfSupDataSample
from mmselfsup.utils import get_module_device
from ..builder import (ALGORITHMS, build_backbone, build_head, build_loss,
                       build_neck)
from .base import BaseModel


@ALGORITHMS.register_module()
class CAE(BaseModel):
    """CAE.

    Implementation of `Context Autoencoder for Self-Supervised Representation
    Learning <https://arxiv.org/abs/2202.03026>`_.

    Args:
        backbone (Dict, optional): Config dict for encoder. Defaults to None.
        neck (Dict, optional): Config dict for encoder. Defaults to None.
        head (Dict, optional): Config dict for head. Defaults to None.
        loss (Dict, optional): Config dict for loss. Defaults to None.
        base_momentum (float): The base momentum coefficient for the target
            network. Defaults to 0.0.
        preprocess_cfg (Dict, optional): Config to preprocess images.
            Defaults to None.
        init_cfg (Union[List[Dict], Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[Dict] = None,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 loss: Optional[Dict] = None,
                 base_momentum: float = 0.0,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[List[Dict], Dict]] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        self.teacher = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        assert loss is not None
        self.loss = build_loss(loss)

        self.momentum = base_momentum

    def init_weights(self) -> None:
        super().init_weights()
        self._init_teacher()

    def _init_teacher(self) -> None:
        # init the weights of teacher with those of backbone
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

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        return self.backbone(inputs[0], mask)

    def forward_train(self, inputs: List[torch.Tensor],
                      data_samples: List[SelfSupDataSample],
                      **kwargs) -> Dict[str, torch.Tensor]:

        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])

        mask = mask.flatten(1).to(torch.bool)

        unmasked = self.backbone(inputs[0], mask)

        # get the latent prediction for the masked patches
        with torch.no_grad():
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
        target = self.head(inputs[1], mask)
        loss_main, loss_align = self.loss(logits, target, latent_pred,
                                          latent_target)
        losses = dict()

        losses['loss'] = loss_main + loss_align
        losses['main'] = loss_main
        losses['align'] = loss_align
        return losses

    def preprocss_data(
            self,
            data: List[Dict]) -> Tuple[List[torch.Tensor], SelfSupDataSample]:
        """Process input data during training, testing or extracting.

        This function overwrites the defaults function in BaseModel by
        normalizing img_target with dalle style normalization.

        Args:
            data (List[Dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple:  It should contain 2 item.
            - batch_images (List[torch.Tensor]): The batch image tensor.
            - data_samples (List[SelfSupDataSample], Optional): The Data
            Samples. It usually includes information such as
            `gt_label`. Return None If the input data does not
            contain `data_sample`.
        """
        # data_['inputs] is a list
        images = [data_['inputs'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        device = get_module_device(self)
        data_samples = [data_sample.to(device) for data_sample in data_samples]
        images = [[img_.to(device) for img_ in img] for img in images]

        # convert images to rgb
        if self.to_rgb and images[0][0].size(0) == 3:
            images = [[img_[[2, 1, 0], ...] for img_ in img] for img in images]

        # normalize images
        images = [[(img[0] - self.mean_norm) / self.std_norm,
                   img[1] * 0.8 + 0.1] for img in images]

        # reconstruct images into several batches. For example, SimCLR needs
        # two crops for each image, and this code snippet will convert images
        # into two batches, each containing one crop of an image.
        batch_images = []
        for i in range(len(images[0])):
            cur_batch = [img[i] for img in images]
            batch_images.append(torch.stack(cur_batch))

        return batch_images, data_samples
