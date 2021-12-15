# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule


class MultiPooling(BaseModule):
    """Pooling layers for features from multiple depth.

    Args:
        pool_type (str): Pooling type for the feature map. Options are
            'adaptive' and 'specified'. Defaults to 'adaptive'.
        in_indices (Sequence[int]): Output from which backbone stages.
            Defaults to (0, ).
        backbone (str): The selected backbone. Defaults to 'resnet50'.
    """

    POOL_PARAMS = {
        'resnet50': [
            dict(kernel_size=10, stride=10, padding=4),
            dict(kernel_size=16, stride=8, padding=0),
            dict(kernel_size=13, stride=5, padding=0),
            dict(kernel_size=8, stride=3, padding=0),
            dict(kernel_size=6, stride=1, padding=0)
        ]
    }
    POOL_SIZES = {'resnet50': [12, 6, 4, 3, 2]}
    POOL_DIMS = {'resnet50': [9216, 9216, 8192, 9216, 8192]}

    def __init__(self,
                 pool_type='adaptive',
                 in_indices=(0, ),
                 backbone='resnet50'):
        super(MultiPooling, self).__init__()
        assert pool_type in ['adaptive', 'specified']
        assert backbone == 'resnet50', 'Now only support resnet50.'
        if pool_type == 'adaptive':
            self.pools = nn.ModuleList([
                nn.AdaptiveAvgPool2d(self.POOL_SIZES[backbone][i])
                for i in in_indices
            ])
        else:
            self.pools = nn.ModuleList([
                nn.AvgPool2d(**self.POOL_PARAMS[backbone][i])
                for i in in_indices
            ])

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return [p(xx) for p, xx in zip(self.pools, x)]
