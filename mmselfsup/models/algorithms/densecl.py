# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class DenseCL(BaseModel):
    """DenseCL.

    Implementation of `Dense Contrastive Learning for Self-Supervised Visual
    Pre-Training <https://arxiv.org/abs/2011.09157>`_.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL>`_.
    The loss_lambda warmup is in `core/hooks/densecl_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
        loss_lambda (float): Loss weight for the single and dense contrastive
            loss. Defaults to 0.5.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 loss_lambda=0.5,
                 init_cfg=None,
                 **kwargs):
        super(DenseCL, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum
        self.loss_lambda = loss_lambda

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        # create the second queue for dense output
        self.register_buffer('queue2', torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer('queue2_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
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

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        """Update queue2."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        im_q = img[0]
        im_k = img[1]
        # compute query features
        q_b = self.encoder_q[0](im_q)  # backbone features
        q, q_grid, q2 = self.encoder_q[1](q_b)  # queries: NxC; NxCxS^2
        q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)

        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k_b = self.encoder_k[0](im_k)  # backbone features
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            k = nn.functional.normalize(k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = batch_unshuffle_ddp(k2, idx_unshuffle)
            k_grid = batch_unshuffle_ddp(k_grid, idx_unshuffle)
            k_b = batch_unshuffle_ddp(k_b, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # feat point set sim
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1]  # NxS^2

        indexed_k_grid = torch.gather(k_grid, 2,
                                      densecl_sim_ind.unsqueeze(1).expand(
                                          -1, k_grid.size(1), -1))  # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1)  # NxS^2

        # dense positive logits: NS^2X1
        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1)

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        # dense negative logits: NS^2xK
        l_neg_dense = torch.einsum(
            'nc,ck->nk', [q_grid, self.queue2.clone().detach()])

        loss_single = self.head(l_pos, l_neg)['loss']
        loss_dense = self.head(l_pos_dense, l_neg_dense)['loss']

        losses = dict()
        losses['loss_single'] = loss_single * (1 - self.loss_lambda)
        losses['loss_dense'] = loss_dense * self.loss_lambda

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)

        return losses

    def forward_test(self, img, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input of two concatenated images of shape
                (N, 2, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict(Tensor): A dictionary of normalized output features.
        """
        im_q = img.contiguous()
        # compute query features
        # _, q_grid, _ = self.encoder_q(im_q)
        q_grid = self.extract_feat(im_q)[0]
        q_grid = q_grid.view(q_grid.size(0), q_grid.size(1), -1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        return None, q_grid, None
