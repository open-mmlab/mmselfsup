# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseDataSource
from .cifar import CIFAR10, CIFAR100
from .image_list import ImageList
from .imagenet import ImageNet

__all__ = ['BaseDataSource', 'CIFAR10', 'CIFAR100', 'ImageList', 'ImageNet']
