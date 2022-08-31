# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models.backbones.resnet import ResLayer
from mmcls.models.backbones.resnext import Bottleneck

from mmselfsup.registry import MODELS
from .resnet import ResNet


@MODELS.register_module()
class ResNeXt(ResNet):
    """ResNeXt backbone.

    Please refer to the `paper <https://arxiv.org/abs/1611.05431>`__ for
    details.

    As the behavior of forward function in MMSelfSup is different from
    MMCls, we register our own ResNeXt, inheriting from
    `mmselfsup.model.backbone.ResNet`.

    Args:
        depth (int): Network depth, from {50, 101, 152}.
        groups (int): Groups of conv2 in Bottleneck. Defaults to 32.
        width_per_group (int): Width per group of conv2 in Bottleneck.
            Defaults to 4.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_channels (int): Output channels of the stem layer. Defaults to 64.
        num_stages (int): Stages of the network. Defaults to 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Defaults to ``(3, )``.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Defaults to False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        conv_cfg (dict | None): The config dict for conv layers.
            Defaults to None.
        norm_cfg (dict): The config dict for norm layers.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Defaults to False.

    Example:
        >>> from mmselfsup.models import ResNeXt
        >>> import torch
        >>> self = ResNeXt(depth=50)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 groups: int = 32,
                 width_per_group: int = 4,
                 **kwargs) -> None:
        self.groups = groups
        self.width_per_group = width_per_group
        super().__init__(depth=depth, **kwargs)

    def make_res_layer(self, **kwargs) -> nn.Module:
        """Redefine the function for ResNeXt related args."""
        return ResLayer(
            groups=self.groups,
            width_per_group=self.width_per_group,
            base_channels=self.base_channels,
            **kwargs)
