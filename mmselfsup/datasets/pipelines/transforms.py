# Copyright (c) OpenMMLab. All rights reserved.
import inspect

import numpy as np
import torch
from mmcv.utils import build_from_cfg
from PIL import Image, ImageFilter
from timm.data import create_transform
from torchvision import transforms as _transforms

from ..builder import PIPELINES

# register all existing transforms in torchvision
_EXCLUDED_TRANSFORMS = ['GaussianBlur']
for m in inspect.getmembers(_transforms, inspect.isclass):
    if m[0] not in _EXCLUDED_TRANSFORMS:
        PIPELINES.register_module(m[1])


@PIPELINES.register_module()
class RandomAug(object):
    """RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation
    with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    This code is borrowed from <https://github.com/pengzhiliang/MAE-pytorch>
    """

    def __init__(self,
                 input_size=None,
                 color_jitter=None,
                 auto_augment=None,
                 interpolation=None,
                 re_prob=None,
                 re_mode=None,
                 re_count=None,
                 mean=None,
                 std=None):

        self.trans = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            mean=mean,
            std=std,
        )

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)
        self.prob = p

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'prob = {self.prob}'
        return repr_str


# custom transforms
@PIPELINES.register_module()
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise).

    Args:
        alphastd (float, optional): The parameter for Lighting.
            Defaults to 0.1.
    """

    _IMAGENET_PCA = {
        'eigval':
        torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec':
        torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self, alphastd=0.1):
        self.alphastd = alphastd
        self.eigval = self._IMAGENET_PCA['eigval']
        self.eigvec = self._IMAGENET_PCA['eigvec']

    def __call__(self, img):
        assert isinstance(img, torch.Tensor), \
            f'Expect torch.Tensor, got {type(img)}'
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'alphastd = {self.alphastd}'
        return repr_str


@PIPELINES.register_module()
class GaussianBlur(object):
    """GaussianBlur augmentation refers to `SimCLR.

    <https://arxiv.org/abs/2002.05709>`_.

    Args:
        sigma_min (float): The minimum parameter of Gaussian kernel std.
        sigma_max (float): The maximum parameter of Gaussian kernel std.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, sigma_min, sigma_max, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0,1], got {p} instead.'
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'sigma_min = {self.sigma_min}, '
        repr_str += f'sigma_max = {self.sigma_max}, '
        repr_str += f'prob = {self.prob}'
        return repr_str


@PIPELINES.register_module()
class Solarization(object):
    """Solarization augmentation refers to `BYOL.

    <https://arxiv.org/abs/2006.07733>`_.

    Args:
        threshold (float, optional): The solarization threshold.
            Defaults to 128.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, threshold=128, p=0.5):
        assert 0 <= p <= 1.0, \
            f'The prob should be in range [0, 1], got {p} instead.'

        self.threshold = threshold
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        img = np.array(img)
        img = np.where(img < self.threshold, img, 255 - img)
        return Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'threshold = {self.threshold}, '
        repr_str += f'prob = {self.prob}'
        return repr_str
