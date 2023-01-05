# Copyright (c) OpenMMLab. All rights reserved.
from .beit_vit import BEiTViT
from .cae_vit import CAEViT
from .mae_vit import MAEViT
from .maskfeat_vit import MaskFeatViT
<<<<<<< HEAD
from .milan_vit import MILANViT
from .mocov3_vit import MoCoV3ViT
from .resnet import ResNet, ResNetSobel, ResNetV1d
=======
from .mim_cls_vit import MIMVisionTransformer
from .resnet import ResNet, ResNetV1d
>>>>>>> upstream/master
from .resnext import ResNeXt
from .simmim_swin import SimMIMSwinTransformer

__all__ = [
<<<<<<< HEAD
    'ResNet', 'ResNetSobel', 'ResNetV1d', 'ResNeXt', 'MAEViT', 'MoCoV3ViT',
    'SimMIMSwinTransformer', 'CAEViT', 'MaskFeatViT', 'BEiTViT', 'MILANViT'
=======
    'ResNet', 'ResNetV1d', 'ResNeXt', 'MAEViT', 'MIMVisionTransformer',
    'VisionTransformer', 'SimMIMSwinTransformer', 'CAEViT', 'MaskFeatViT'
>>>>>>> upstream/master
]
