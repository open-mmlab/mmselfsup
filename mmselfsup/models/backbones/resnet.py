# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
from mmcls.models.backbones import ResNet as _ResNet
from mmcls.models.backbones.resnet import BasicBlock, Bottleneck

from mmselfsup.registry import MODELS
from ..utils import Sobel


@MODELS.register_module()
class ResNet(_ResNet):
    """ResNet backbone.

    Please refer to the `paper <https://arxiv.org/abs/1512.03385>`__ for
    details.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_channels (int): Output channels of the stem layer. Defaults to 64.
        base_channels (int): Middle channels of the first stage.
            Defaults to 64.
        num_stages (int): Stages of the network. Defaults to 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to ``(4, )``.
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
        Probability of the path to be zeroed. Defaults to 0.1
    Example:
        >>> from mmselfsup.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth: int,
                 in_channels: int = 3,
                 stem_channels: int = 64,
                 base_channels: int = 64,
                 expansion: Optional[int] = None,
                 num_stages: int = 4,
                 strides: Tuple[int] = (1, 2, 2, 2),
                 dilations: Tuple[int] = (1, 1, 1, 1),
                 out_indices: Tuple[int] = (4, ),
                 style: str = 'pytorch',
                 deep_stem: bool = False,
                 avg_down: bool = False,
                 frozen_stages: int = -1,
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(
                     type='BN', requires_grad=True),
                 norm_eval: bool = False,
                 with_cp: bool = False,
                 zero_init_residual: bool = False,
                 init_cfg: Optional[dict] = [
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate: float = 0.0,
                 **kwargs) -> None:
        # As the meaning of the `out_indices` is different between MMCls and
        # MMSelfSup (Number 3 indicates the last stage output in MMCls but
        # number 4 in MMSelfSup) and there is a sanity check in MMCls ResNet
        # init function, we use a fake input of `out_indices` to pass it.
        temp_out_indices = out_indices
        out_indices = (3, )
        super().__init__(
            depth=depth,
            in_channels=in_channels,
            stem_channels=stem_channels,
            base_channels=base_channels,
            expansion=expansion,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            init_cfg=init_cfg,
            drop_path_rate=drop_path_rate,
            **kwargs)
        self.out_indices = temp_out_indices
        assert max(out_indices) < num_stages + 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward function.

        As the behavior of forward function in MMSelfSup is different from
        MMCls, we rewrite the forward function. MMCls does not output the
        feature map from the 'stem' layer, which will be used for downstream
        evaluation.
        """
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)  # r50: 64x128x128
        outs = []
        if 0 in self.out_indices:
            outs.append(x)
        x = self.maxpool(x)  # r50: 64x56x56
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i + 1 in self.out_indices:
                outs.append(x)
        # r50: 1-256x56x56; 2-512x28x28; 3-1024x14x14; 4-2048x7x7
        return tuple(outs)


@MODELS.register_module()
class ResNetSobel(ResNet):
    """ResNet with Sobel layer.

    This variant is used in clustering-based methods like DeepCluster to avoid
    color shortcut.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(in_channels=2, **kwargs)
        self.sobel_layer = Sobel()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward function."""
        x = self.sobel_layer(x)
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)  # r50: 64x128x128
        outs = []
        if 0 in self.out_indices:
            outs.append(x)
        x = self.maxpool(x)  # r50: 64x56x56
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i + 1 in self.out_indices:
                outs.append(x)
        # r50: 1-256x56x56; 2-512x28x28; 3-1024x14x14; 4-2048x7x7
        return tuple(outs)


@MODELS.register_module()
class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
