# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule


class MultiPrototypes(BaseModule):
    """Multi-prototypes for SwAV head.

    Args:
        output_dim (int): The output dim from SwAV neck.
        num_prototypes (list[int]): The number of prototypes needed.
    """

    def __init__(self, output_dim, num_prototypes):
        super(MultiPrototypes, self).__init__()
        assert isinstance(num_prototypes, list)
        self.num_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_module('prototypes' + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.num_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out
