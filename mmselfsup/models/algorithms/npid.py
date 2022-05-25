# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmselfsup.core import SelfSupDataSample
from ..builder import (ALGORITHMS, build_backbone, build_head, build_loss,
                       build_memory, build_neck)
from .base import BaseModel


@ALGORITHMS.register_module()
class NPID(BaseModel):
    """NPID.

    Implementation of `Unsupervised Feature Learning via Non-parametric
    Instance Discrimination <https://arxiv.org/abs/1805.01978>`_.

    Args:
        backbone (Dict): Config dict for module of backbone.
        neck (Dict, optional): Config dict for module of deep features to
            compact feature vectors. Defaults to None.
        head (Dict, optional): Config dict for module of head functions.
            Defaults to None.
        loss (dict): Config dict for module of loss functions.
            Defaults to None.
        memory_bank (Dict, optional): Config dict for module of memory banks.
            Defaults to None.
        neg_num (int): Number of negative samples for each image.
            Defaults to 65536.
        ensure_neg (bool): If False, there is a small probability
            that negative samples contain positive ones. Defaults to False.
        preprocess_cfg (Dict, optional): Config to preprocess images.
            Defaults to None.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Dict,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 loss: Optional[Dict] = None,
                 memory_bank: Optional[Dict] = None,
                 neg_num: Optional[int] = 65536,
                 ensure_neg: Optional[bool] = False,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)
        assert loss is not None
        self.loss = build_loss(loss)
        assert memory_bank is not None
        self.memory_bank = build_memory(memory_bank)

        self.neg_num = neg_num
        self.ensure_neg = ensure_neg

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
            Dict[str, Tensor]: A dictionary of loss components.
        """
        feature = self.extract_feat(inputs[0])
        idx = [data_sample.idx for data_sample in data_samples]
        idx = torch.cat(idx)
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

        logits, labels = self.head(pos_logits, neg_logits)
        loss = self.loss(logits, labels)
        losses = dict(loss=loss)
        # update memory bank
        with torch.no_grad():
            self.memory_bank.update(idx, feature.detach())

        return losses
