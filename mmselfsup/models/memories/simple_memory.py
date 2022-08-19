# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmengine.dist import all_gather, get_dist_info
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS
from mmselfsup.utils import AliasMethod


@MODELS.register_module()
class SimpleMemory(BaseModule):
    """Simple feature memory bank.

    This module includes the memory bank that stores running average
    features of all samples in the dataset. It is used in algorithms
    like NPID.

    Args:
        length (int): Number of features stored in the memory bank.
        feat_dim (int): Dimension of stored features.
        momentum (float): Momentum coefficient for updating features.
    """

    def __init__(self, length: int, feat_dim: int, momentum: float,
                 **kwargs) -> None:
        super().__init__()
        self.rank, self.num_replicas = get_dist_info()
        self.register_buffer('feature_bank', torch.randn(length, feat_dim))
        self.feature_bank = nn.functional.normalize(self.feature_bank)
        self.momentum = momentum
        self.multinomial = AliasMethod(torch.ones(length))

    def update(self, idx: torch.Tensor, feature: torch.Tensor) -> None:
        """Update features in the memory bank.

        Args:
            idx (torch.Tensor): Indices for the batch of features.
            feature (torch.Tensor): Batch of features.
        """
        feature_norm = nn.functional.normalize(feature)
        idx, feature_norm = self._gather(idx, feature_norm)
        feature_old = self.feature_bank[idx, ...]
        feature_new = (1 - self.momentum) * feature_old + \
            self.momentum * feature_norm
        feature_new_norm = nn.functional.normalize(feature_new)
        self.feature_bank[idx, ...] = feature_new_norm

    def _gather(self, idx: torch.Tensor,
                feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather indices and features.

        Args:
            idx (torch.Tensor): Indices for the batch of features.
            feature (torch.Tensor): Batch of features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Gathered information.
                - idx_gathered: Gathered indices.
                - feature_gathered: Gathered features.
        """
        idx_gathered = all_gather(idx)
        feature_gathered = all_gather(feature)
        idx_gathered = torch.cat(idx_gathered, dim=0)
        feature_gathered = torch.cat(feature_gathered, dim=0)
        return idx_gathered, feature_gathered
