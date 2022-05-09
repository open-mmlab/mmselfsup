# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from PIL import Image

from mmselfsup.datasets.pipelines import (
    BEiTMaskGenerator, RandomResizedCropAndInterpolationWithTwoPic,
    SimMIMMaskGenerator)


def test_simmim_mask_gen():
    transform = dict(
        input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6)

    img = torch.rand((3, 192, 192))
    results = {'img': img}
    module = SimMIMMaskGenerator(**transform)

    results = module(results)

    # test transform
    assert list(results['img'].shape) == [3, 192, 192]
    assert list(results['mask'].shape) == [48, 48]

    # test repr
    assert isinstance(str(module), str)


def test_beit_mask_gen():
    transform = dict(
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=None,
        min_num_patches=16)
    module = BEiTMaskGenerator(**transform)
    results = {}

    results = module(results)

    # test transform
    assert list(results['mask'].shape) == [14, 14]

    # test repr
    assert isinstance(str(module), str)


def test_random_resize_crop_with_two_pic():
    transform = dict(
        size=224,
        second_size=112,
        interpolation='bicubic',
        second_interpolation='lanczos',
        scale=(0.08, 1.0))
    module = RandomResizedCropAndInterpolationWithTwoPic(**transform)
    fake_input = torch.rand((224, 224, 3)).numpy().astype(np.uint8)
    fake_input = Image.fromarray(fake_input)

    results = {'img': fake_input}
    results = module(results)

    # test transform
    assert list(results['img'].size) == [224, 224]
    assert list(results['target_img'].size) == [112, 112]

    # test repr
    assert isinstance(str(module), str)
