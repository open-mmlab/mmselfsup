# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmengine.model import BaseModule


class MultiPrototypes(BaseModule):
    """Multi-prototypes for SwAV head.

    Args:
        output_dim (int): The output dim from SwAV neck.
        num_prototypes (List[int]): The number of prototypes needed.
    """

    def __init__(self, output_dim: int, num_prototypes: List[int]) -> None:
        super().__init__()
        assert isinstance(num_prototypes, list)
        self.num_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_module('prototypes' + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run forward for every prototype."""
        out = []
        for i in range(self.num_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out
