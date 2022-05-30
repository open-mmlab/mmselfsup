# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import ExponentialMovingAverage

from mmselfsup.core import SelfSupDataSample
from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import (ALGORITHMS, build_backbone, build_head, build_loss,
                       build_neck)
from .base import BaseModel


@ALGORITHMS.register_module()
class MoCo(BaseModel):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (Dict): Config dict for module of backbone.
        neck (Dict): Config dict for module of deep features to compact feature
            vectors.
        head (Dict): Config dict for module of head functions.
        loss (Dict): Config dict for module of loss functions.
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
                 backbone: Dict,
                 neck: Dict,
                 head: Dict,
                 loss: Dict,
                 queue_len: Optional[int] = 65536,
                 feat_dim: Optional[int] = 128,
                 momentum: Optional[float] = 0.999,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[List[Dict], Dict]] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        assert head is not None
        self.head = build_head(head)
        assert loss is not None
        self.loss = build_loss(loss)

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

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        x = self.backbone(inputs[0])
        return x

    def forward_train(self, inputs: List[torch.Tensor],
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
        im_q = inputs[0]
        im_k = inputs[1]
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

        logits, labels = self.head(l_pos, l_neg)
        loss = self.loss(logits, labels)

        # update the queue
        self._dequeue_and_enqueue(k)
        losses = dict(loss=loss)
        return losses
