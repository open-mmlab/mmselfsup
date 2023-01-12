# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS
import torch.nn as nn

@MODELS.register_module()
class DINONeck(BaseModule):

    def __init__(self) -> None:
        super().__init__(self,
                         in_dim,
                         out_dim,
                         use_bn=False,
                         norm_last_layer=True,
                         nlayers=3,
                         hidden_dim=2048,
                         bottleneck_dim=256)
        # TODO: implement the initialization function
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    def forward(self):
        """Forward function of DINO Neck."""
        # TODO: implement the forward pass of neck here
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
