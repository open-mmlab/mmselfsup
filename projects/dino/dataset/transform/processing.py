# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import BaseTransform, RandomFlip, RandomApply, RandomGrayscale, Compose  # noqa: E501
from mmcls.datasets.transforms import ColorJitter, RandomResizedCrop
from mmselfsup.datasets.transforms import RandomGaussianBlur, RandomSolarize

from mmselfsup.registry import TRANSFORMS


@TRANSFORMS.register_module()
class DINOMultiCrop(BaseTransform):

    def __init__(self, global_crops_scale: int, local_crops_scale: int,
                 local_crop_number: int) -> None:
        super().__init__()

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        flip_and_color_jitter = Compose([
            RandomFlip(prob=0.5, direction='horizontal'),
            RandomApply([
                ColorJitter(
                    brighness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ],
                        p=0.8),
            RandomGrayscale(p=0.2),
        ])

        self.global_transform_1 = Compose([
            RandomResizedCrop(
                224, scale=global_crops_scale, interpolation='bicubic'),
            flip_and_color_jitter,
            RandomGaussianBlur(prob=1.0),
        ])

        self.global_transform_2 = Compose([
            RandomResizedCrop(
                224, scale=global_crops_scale, interpolation='bicubic'),
            flip_and_color_jitter,
            RandomSolarize(prob=1.0),
            RandomSolarize(prob=0.2),
        ])

        self.local_crops_number = local_crop_number
        self.local_transform = Compose([
            RandomResizedCrop(
                96, scale=local_crops_scale, interpolation='bicubic'),
            flip_and_color_jitter,
            RandomGaussianBlur(prob=1.0),
        ])

    def transform(self, results: dict) -> dict:
        img = results['img']
        crops = []
        crops.append(self.global_transform_1(img))
        crops.append(self.global_transform_2(img))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(img))
        results['img'] = crops
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(global_crops_scale = {self.global_crops_scale}, '
        repr_str += f'local_crops_scale = {self.local_crops_scale}, '
        repr_str += f'local_crop_number = {self.local_crops_number})'
        return repr_str