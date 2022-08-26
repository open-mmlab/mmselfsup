# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class NPID(BaseModel):
    """NPID.

    Implementation of `Unsupervised Feature Learning via Non-parametric
    Instance Discrimination <https://arxiv.org/abs/1805.01978>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to
            compact feature vectors.
        head (dict): Config dict for module of head functions.
        memory_bank (dict): Config dict for module of memory bank.
        neg_num (int): Number of negative samples for each image.
            Defaults to 65536.
        ensure_neg (bool): If False, there is a small probability
            that negative samples contain positive ones. Defaults to False.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
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
                 memory_bank: dict,
                 neg_num: int = 65536,
                 ensure_neg: bool = False,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        assert memory_bank is not None
        self.memory_bank = MODELS.build(memory_bank)

        self.neg_num = neg_num
        self.ensure_neg = ensure_neg

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        x = self.backbone(inputs[0])
        return x

    def loss(self, inputs: List[torch.Tensor],
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
        feature = self.backbone(inputs[0])
        idx = [data_sample.sample_idx.value for data_sample in data_samples]
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

        loss = self.head(pos_logits, neg_logits)
        losses = dict(loss=loss)
        # update memory bank
        with torch.no_grad():
            self.memory_bank.update(idx, feature.detach())

        return losses
