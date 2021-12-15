# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import (ALGORITHMS, build_backbone, build_head, build_memory,
                       build_neck)
from .base import BaseModel


@ALGORITHMS.register_module()
class NPID(BaseModel):
    """NPID.

    Implementation of `Unsupervised Feature Learning via Non-parametric
    Instance Discrimination <https://arxiv.org/abs/1805.01978>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        memory_bank (dict): Config dict for module of memory banks.
            Defaults to None.
        neg_num (int): Number of negative samples for each image.
            Defaults to 65536.
        ensure_neg (bool): If False, there is a small probability
            that negative samples contain positive ones. Defaults to False.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 memory_bank=None,
                 neg_num=65536,
                 ensure_neg=False,
                 init_cfg=None):
        super(NPID, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        assert memory_bank is not None
        self.memory_bank = build_memory(memory_bank)

        self.neg_num = neg_num
        self.ensure_neg = ensure_neg

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
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            idx (Tensor): Index corresponding to each image.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        feature = self.extract_feat(img)
        idx = idx.cuda()
        if self.with_neck:
            feature = self.neck(feature)[0]
        feature = nn.functional.normalize(feature)  # BxC
        bs, feat_dim = feature.shape[:2]
        neg_idx = self.memory_bank.multinomial.draw(bs * self.neg_num)
        if self.ensure_neg:
            neg_idx = neg_idx.view(bs, -1)
            while True:
                wrong = (neg_idx == idx.view(-1, 1))
                if wrong.sum().item() > 0:
                    neg_idx[wrong] = self.memory_bank.multinomial.draw(
                        wrong.sum().item())
                else:
                    break
            neg_idx = neg_idx.flatten()

        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs, self.neg_num,
                                                    feat_dim)  # BxKxC

        pos_logits = torch.einsum('nc,nc->n',
                                  [pos_feat, feature]).unsqueeze(-1)
        neg_logits = torch.bmm(neg_feat, feature.unsqueeze(2)).squeeze(2)

        losses = self.head(pos_logits, neg_logits)

        # update memory bank
        with torch.no_grad():
            self.memory_bank.update(idx, feature.detach())

        return losses
