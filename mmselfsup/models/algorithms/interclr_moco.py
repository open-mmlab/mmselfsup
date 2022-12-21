# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import (ALGORITHMS, build_backbone, build_head, build_memory,
                       build_neck)
from .base import BaseModel


@ALGORITHMS.register_module()
class InterCLRMoCo(BaseModel):
    """MoCo-InterCLR.

    Official implementation of `Delving into Inter-Image Invariance for
    Unsupervised Visual Representations <https://arxiv.org/abs/2008.11702>`_.
    The clustering operation is in `core/hooks/interclr_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
        memory_bank (dict): Config dict for module of memory banks.
            Defaults to None.
        online_labels (bool): Whether to use online labels. Defaults to True.
        neg_num (int): Number of negative samples for inter-image branch.
            Defaults to 16384.
        neg_sampling (str): Negative sampling strategy. Support 'hard',
            'semihard', 'random', 'semieasy'. Defaults to 'semihard'.
        semihard_neg_pool_num (int): Number of negative samples for semi-hard
            nearest neighbor pool. Defaults to 128000.
        semieasy_neg_pool_num (int): Number of negative samples for semi-easy
            nearest neighbor pool. Defaults to 128000.
        intra_cos_marign_loss (bool): Whether to use a cosine margin for
            intra-image branch. Defaults to False.
        intra_cos_marign (float): Intra-image cosine margin. Defaults to 0.
        intra_arc_marign_loss (bool): Whether to use an arc margin for
            intra-image branch. Defaults to False.
        intra_arc_marign (float): Intra-image arc margin. Defaults to 0.
        inter_cos_marign_loss (bool): Whether to use a cosine margin for
            inter-image branch. Defaults to True.
        inter_cos_marign (float): Inter-image cosine margin. Defaults to -0.5.
        inter_arc_marign_loss (bool): Whether to use an arc margin for
            inter-image branch. Defaults to False.
        inter_arc_marign (float): Inter-image arc margin. Defaults to 0.
        intra_loss_weight (float): Loss weight for intra-image branch.
            Defaults to 0.75.
        inter_loss_weight (float): Loss weight for inter-image branch.
            Defaults to 0.25.
        share_neck (bool): Whether to share the neck for intra- and inter-image
            branches. Defaults to True.
        num_classes (int): Number of clusters. Defaults to 10000.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 memory_bank=None,
                 online_labels=True,
                 neg_num=16384,
                 neg_sampling='semihard',
                 semihard_neg_pool_num=128000,
                 semieasy_neg_pool_num=128000,
                 intra_cos_marign_loss=False,
                 intra_cos_margin=0,
                 intra_arc_marign_loss=False,
                 intra_arc_margin=0,
                 inter_cos_marign_loss=True,
                 inter_cos_margin=-0.5,
                 inter_arc_marign_loss=False,
                 inter_arc_margin=0,
                 intra_loss_weight=0.75,
                 inter_loss_weight=0.25,
                 share_neck=True,
                 num_classes=10000,
                 init_cfg=None,
                 **kwargs):
        super(InterCLRMoCo, self).__init__(init_cfg)
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        if not share_neck:
            self.inter_neck_q = build_neck(neck)
            self.inter_neck_k = build_neck(neck)
        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        self.head = build_head(head)
        self.memory_bank = build_memory(memory_bank)

        # moco params
        self.queue_len = queue_len
        self.momentum = momentum

        # interclr params
        self.online_labels = online_labels
        self.neg_num = neg_num
        self.neg_sampling = neg_sampling
        self.semihard_neg_pool_num = semihard_neg_pool_num
        self.semieasy_neg_pool_num = semieasy_neg_pool_num
        self.intra_cos = intra_cos_marign_loss
        self.intra_cos_margin = intra_cos_margin
        self.intra_arc = intra_arc_marign_loss
        self.intra_arc_margin = intra_arc_margin
        self.inter_cos = inter_cos_marign_loss
        self.inter_cos_margin = inter_cos_margin
        self.inter_arc = inter_arc_marign_loss
        self.inter_arc_margin = inter_arc_margin
        self.intra_loss_weight = intra_loss_weight
        self.inter_loss_weight = inter_loss_weight
        self.share_neck = share_neck
        self.num_classes = num_classes

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def init_weights(self):
        """Initialize base_encoder with init_cfg defined in backbone."""
        super(InterCLRMoCo, self).init_weights()

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        if not self.share_neck:
            for param_q, param_k in zip(self.inter_neck_q.parameters(),
                                        self.inter_neck_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)
        if not self.share_neck:
            for param_q, param_k in zip(self.inter_neck_q.parameters(),
                                        self.inter_neck_k.parameters()):
                param_k.data = param_k.data * self.momentum + \
                               param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # normalize
        keys = nn.functional.normalize(keys, dim=1)
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def contrast_intra(self, q, k):
        """Intra-image invariance learning.

        Args:
            q (Tensor): Query features with shape (N, C).
            k (Tensor): Key features with shape (N, C).

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        pos_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        neg_logits = torch.einsum('nc,ck->nk',
                                  [q, self.queue.clone().detach()])

        # use cosine margin
        if self.intra_cos:
            cosine = pos_logits.clone()
            phi = cosine - self.intra_cos_margin
            pos_logits.copy_(phi)
        # use arc margin
        if self.intra_arc:
            cosine = pos_logits.clone()
            sine = torch.sqrt((1.0 - cosine**2).clamp(0, 1))
            phi = cosine * math.cos(self.intra_arc_margin) - sine * math.sin(
                self.intra_arc_margin)
            if self.intra_arc_margin < 0:
                phi = torch.where(
                    cosine < math.cos(self.intra_arc_margin), phi, cosine +
                    math.sin(self.intra_arc_margin) * self.intra_arc_margin)
            else:
                phi = torch.where(
                    cosine > math.cos(math.pi - self.intra_arc_margin), phi,
                    cosine - math.sin(math.pi - self.intra_arc_margin) *
                    self.intra_arc_margin)
            pos_logits.copy_(phi)

        losses = self.head(pos_logits, neg_logits)

        return losses

    def contrast_inter(self, q, idx):
        """Inter-image invariance learning.

        Args:
            q (Tensor): Query features with shape (N, C).
            idx (Tensor): Index corresponding to each query.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # normalize
        feat_norm = nn.functional.normalize(q, dim=1)
        bs, feat_dim = feat_norm.shape[:2]
        # positive sampling
        pos_label = self.memory_bank.label_bank[idx.cpu()]
        pos_idx_list = []
        for i, l in enumerate(pos_label):
            pos_idx_pool = torch.where(
                self.memory_bank.label_bank == l)[0]  # positive index pool
            pos_i = torch.zeros(
                1, dtype=torch.long).random_(0, pos_idx_pool.size(0))
            pos_idx_list.append(pos_idx_pool[pos_i])
        pos_idx = torch.cuda.LongTensor(pos_idx_list)
        # negative sampling
        if self.neg_sampling == 'random':  # random negative sampling
            pos_label = pos_label.cuda().unsqueeze(1)
            neg_idx = self.memory_bank.multinomial.draw(
                bs * self.neg_num).view(bs, -1)
            while True:
                neg_label = self.memory_bank.label_bank[neg_idx.cpu()].cuda()
                pos_in_neg_idx = (neg_label == pos_label)
                if pos_in_neg_idx.sum().item() > 0:
                    neg_idx[
                        pos_in_neg_idx] = self.memory_bank.multinomial.draw(
                            pos_in_neg_idx.sum().item())
                else:
                    break
        elif self.neg_sampling == 'semihard':  # semihard negative sampling
            pos_label = pos_label.cuda().unsqueeze(1)
            similarity = torch.mm(feat_norm.detach(),
                                  self.memory_bank.feature_bank.permute(1, 0))
            _, neg_I = torch.topk(
                similarity, self.semihard_neg_pool_num, dim=1, sorted=False)
            weights = torch.ones((bs, self.semihard_neg_pool_num),
                                 dtype=torch.float,
                                 device='cuda')
            neg_i = torch.multinomial(weights, self.neg_num)
            neg_idx = torch.gather(neg_I, 1, neg_i)
            while True:
                neg_label = self.memory_bank.label_bank[neg_idx.cpu()].cuda()
                pos_in_neg_idx = (neg_label == pos_label)
                if pos_in_neg_idx.sum().item() > 0:
                    neg_i = torch.multinomial(weights, self.neg_num)
                    neg_idx[pos_in_neg_idx] = torch.gather(
                        neg_I, 1, neg_i)[pos_in_neg_idx]
                else:
                    break
        elif self.neg_sampling == 'semieasy':  # semieasy negative sampling
            pos_label = pos_label.cuda().unsqueeze(1)
            similarity = torch.mm(feat_norm.detach(),
                                  self.memory_bank.feature_bank.permute(1, 0))
            _, neg_I = torch.topk(
                similarity,
                self.semieasy_neg_pool_num,
                dim=1,
                largest=False,
                sorted=False)
            weights = torch.ones((bs, self.semieasy_neg_pool_num),
                                 dtype=torch.float,
                                 device='cuda')
            neg_i = torch.multinomial(weights, self.neg_num)
            neg_idx = torch.gather(neg_I, 1, neg_i)
            while True:
                neg_label = self.memory_bank.label_bank[neg_idx.cpu()].cuda()
                pos_in_neg_idx = (neg_label == pos_label)
                if pos_in_neg_idx.sum().item() > 0:
                    neg_i = torch.multinomial(weights, self.neg_num)
                    neg_idx[pos_in_neg_idx] = torch.gather(
                        neg_I, 1, neg_i)[pos_in_neg_idx]
                else:
                    break
        elif self.neg_sampling == 'hard':  # hard negative sampling
            similarity = torch.mm(feat_norm.detach(),
                                  self.memory_bank.feature_bank.permute(1, 0))
            maximal_cls_size = np.bincount(
                self.memory_bank.label_bank.numpy(),
                minlength=self.num_classes).max().item()
            _, neg_I = torch.topk(
                similarity, self.neg_num + maximal_cls_size, dim=1)
            neg_I = neg_I.cpu()
            neg_label = self.memory_bank.label_bank[neg_I].numpy()
            neg_idx_list = []
            for i, l in enumerate(pos_label):
                pos_in_neg_idx = np.where(neg_label[i] == l)[0]
                if len(pos_in_neg_idx) > 0:
                    neg_idx_pool = torch.from_numpy(
                        np.delete(neg_I[i].numpy(), pos_in_neg_idx))
                else:
                    neg_idx_pool = neg_I[i]
                neg_idx_list.append(neg_idx_pool[:self.neg_num])
            neg_idx = torch.stack(neg_idx_list, dim=0).cuda()
        else:
            raise Exception(
                f'No {self.neg_sampling} negative sampling strategy.')

        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      pos_idx)  # BXC
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx.flatten()).view(
                                          bs, self.neg_num, feat_dim)  # BxKxC

        pos_logits = torch.einsum('nc,nc->n',
                                  [pos_feat, feat_norm]).unsqueeze(-1)
        neg_logits = torch.bmm(neg_feat, feat_norm.unsqueeze(2)).squeeze(2)

        # use cosine margin
        if self.inter_cos:
            cosine = pos_logits.clone()
            phi = cosine - self.inter_cos_margin
            pos_logits.copy_(phi)
        # use arc margin
        if self.inter_arc:
            cosine = pos_logits.clone()
            sine = torch.sqrt((1.0 - cosine**2).clamp(0, 1))
            phi = cosine * math.cos(self.inter_arc_margin) - sine * math.sin(
                self.inter_arc_margin)
            if self.inter_arc_margin < 0:
                phi = torch.where(
                    cosine < math.cos(self.inter_arc_margin), phi, cosine +
                    math.sin(self.inter_arc_margin) * self.inter_arc_margin)
            else:
                phi = torch.where(
                    cosine > math.cos(math.pi - self.inter_arc_margin), phi,
                    cosine - math.sin(math.pi - self.inter_arc_margin) *
                    self.inter_arc_margin)
            pos_logits.copy_(phi)

        losses = self.head(pos_logits, neg_logits)

        return losses

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

    def forward_train(self, img, idx, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
            idx (Tensor): Index corresponding to each image.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        im_q = img[0]
        im_k = img[1]

        # compute query features
        q_b = self.encoder_q[0](im_q)  # backbone features
        q = self.encoder_q[1](q_b)[0]  # queries: NxC
        if not self.share_neck:
            q2 = self.inter_neck_q(q_b)[0]  # inter queries: NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k_b = self.encoder_k[0](im_k)  # backbone features
            k = self.encoder_k[1](k_b)[0]  # keys: NxC
            if not self.share_neck:
                k2 = self.inter_neck_k(k_b)[0]  # inter keys: NxC

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)
            if not self.share_neck:
                k2 = batch_unshuffle_ddp(k2, idx_unshuffle)

        idx = idx.cuda()
        self.memory_bank.broadcast_feature_bank()
        # compute intra loss
        intra_losses = self.contrast_intra(q, k)
        # compute inter loss
        if self.share_neck:
            inter_losses = self.contrast_inter(q, idx)
        else:
            inter_losses = self.contrast_inter(q2, idx)
        losses = dict()
        losses['intra_loss'] = self.intra_loss_weight * intra_losses['loss']
        losses['inter_loss'] = self.inter_loss_weight * inter_losses['loss']

        self._dequeue_and_enqueue(k)

        # update memory bank
        if self.online_labels:
            if self.share_neck:
                change_ratio = self.memory_bank.update_samples_memory(
                    idx, k.detach())
            else:
                change_ratio = self.memory_bank.update_samples_memory(
                    idx, k2.detach())
            losses['change_ratio'] = change_ratio
        else:
            if self.share_neck:
                self.memory_bank.update_simple_memory(idx, k.detach())
            else:
                self.memory_bank.update_simple_memory(idx, k2.detach())

        return losses
