# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import ExponentialMovingAverage

from mmselfsup.core import SelfSupDataSample
from mmselfsup.registry import MODELS
from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from .base import BaseModel


@MODELS.register_module()
class MoCo(BaseModel):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors.
        head (dict): Config dict for module of head functions.
        queue_len (int, optional): Number of negative keys maintained in the
            queue. Defaults to 65536.
        feat_dim (int, optional): Dimension of compact feature vectors.
            Defaults to 128.
        momentum (float, optional): Momentum coefficient for the
            momentum-updated encoder. Defaults to 0.999.
        preprocess_cfg (Dict, optional): Config to preprocess images.
            Defaults to None.
        init_cfg (Dict or list[Dict], optional): Initialization config dict.
            Defaults to None
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 queue_len: Optional[int] = 65536,
                 feat_dim: Optional[int] = 128,
                 momentum: Optional[float] = 0.999,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.encoder_q = nn.Sequential(self.backbone, self.neck)

        # create momentum model
        self.encoder_k = ExponentialMovingAverage(self.encoder_q, 1 - momentum)
        for param_k in self.encoder_k.module.parameters():
            param_k.requires_grad = False

        # create the queue
        self.queue_len = queue_len
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def extract_feat(self, batch_inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        x = self.backbone(batch_inputs[0])
        return x

    def loss(self, batch_inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            batch_inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        im_q = batch_inputs[0]
        im_k = batch_inputs[1]
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self.encoder_k.update_parameters(self.encoder_q)

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        loss = self.head(l_pos, l_neg)
        # update the queue
        self._dequeue_and_enqueue(k)

        losses = dict(loss=loss)
        return losses
