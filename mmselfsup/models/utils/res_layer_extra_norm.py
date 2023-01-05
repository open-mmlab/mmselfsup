# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_norm_layer

try:
    from mmdet.models.backbones import ResNet
    from mmdet.models.builder import SHARED_HEADS
    from mmdet.models.roi_heads.shared_heads.res_layer import ResLayer

    @SHARED_HEADS.register_module()
    class ResLayerExtraNorm(ResLayer):
        """Add extra norm to original ``ResLayer``."""

        def __init__(self, *args, **kwargs):
            super(ResLayerExtraNorm, self).__init__(*args, **kwargs)

            block = ResNet.arch_settings[kwargs['depth']][0]
            self.add_module(
                'norm',
                build_norm_layer(self.norm_cfg,
                                 64 * 2**self.stage * block.expansion)[1])

        def forward(self, x):
            """Forward function."""
            res_layer = getattr(self, f'layer{self.stage + 1}')
            norm = getattr(self, 'norm')
            x = res_layer(x)
            out = norm(x)
            return out

except ImportError:
    pass
