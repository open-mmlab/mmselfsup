# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmcv.utils import build_from_cfg
from PIL import Image

from mmselfsup.datasets.builder import PIPELINES


def test_random_applied_trans():
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8))

    # p=0.5
    transform = dict(
        type='RandomAppliedTrans', transforms=[dict(type='Solarization')])
    module = build_from_cfg(transform, PIPELINES)
    assert isinstance(str(module), str)
    res = module(img)
    assert img.size == res.size

    transform = dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='Solarization')],
        p=0.)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)
    assert img.size == res.size

    # p=1.
    transform = dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='Solarization')],
        p=1.)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)
    assert img.size == res.size


def test_lighting():
    transform = dict(type='Lighting')
    module = build_from_cfg(transform, PIPELINES)
    assert isinstance(str(module), str)
    img = np.ones((224, 224, 3), dtype=np.uint8)

    with pytest.raises(AssertionError):
        res = module(img)

    img = torch.from_numpy(img).float().permute(2, 1, 0)
    res = module(img)
    assert img.size() == res.size()

    transform = dict(type='Lighting', alphastd=0)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)
    assert img.equal(res)


def test_gaussianblur():
    with pytest.raises(AssertionError):
        transform = dict(
            type='GaussianBlur', sigma_min=0.1, sigma_max=1.0, p=-1)
        module = build_from_cfg(transform, PIPELINES)

    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8))

    # p=0.5
    transform = dict(type='GaussianBlur', sigma_min=0.1, sigma_max=1.0)
    module = build_from_cfg(transform, PIPELINES)
    assert isinstance(str(module), str)
    res = module(img)

    transform = dict(type='GaussianBlur', sigma_min=0.1, sigma_max=1.0, p=0.)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)

    transform = dict(type='GaussianBlur', sigma_min=0.1, sigma_max=1.0, p=1.)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)

    assert img.size == res.size


def test_solarization():
    with pytest.raises(AssertionError):
        transform = dict(type='Solarization', p=-1)
        module = build_from_cfg(transform, PIPELINES)

    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8))

    # p=0.5
    transform = dict(type='Solarization')
    module = build_from_cfg(transform, PIPELINES)
    assert isinstance(str(module), str)
    res = module(img)

    transform = dict(type='Solarization', p=0.)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)

    transform = dict(type='Solarization', p=1.)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)

    assert img.size == res.size


def test_randomaug():
    transform = dict(
        type='RandomAug',
        input_size=224,
        color_jitter=None,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))

    img = Image.fromarray(np.uint8(np.ones((224, 224, 3))))

    module = build_from_cfg(transform, PIPELINES)
    res = module(img)

    assert list(res.shape) == [3, 224, 224]

    assert isinstance(str(module), str)


def test_simmim_mask_gen():
    transform = dict(
        type='SimMIMMaskGenerator',
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6)

    img = torch.rand((3, 192, 192))
    module = build_from_cfg(transform, PIPELINES)

    res = module(img)

    assert list(res[0].shape) == [3, 192, 192]
    assert list(res[1].shape) == [48, 48]


def test_beit_mask_gen():
    transform = dict(
        type='BEiTMaskGenerator',
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=None,
        min_num_patches=16)
    fake_image_1 = torch.rand((3, 224, 224))
    fake_image_2 = torch.rand((3, 112, 112))
    module = build_from_cfg(transform, PIPELINES)

    res = module([fake_image_1, fake_image_2])

    assert list(res[0].shape) == [3, 224, 224]
    assert list(res[1].shape) == [3, 112, 112]
    assert list(res[2].shape) == [14, 14]


def test_to_tensor():
    transform = dict(type='ToTensor')
    module = build_from_cfg(transform, PIPELINES)
    fake_img = torch.rand((112, 112, 3)).numpy()
    fake_output_1 = module(fake_img)
    fake_output_2 = module([fake_img, fake_img])
    assert list(fake_output_1.shape) == [3, 112, 112]
    assert len(fake_output_2) == 2


def test_random_resize_crop_with_two_pic():
    transform = dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
        size=224,
        second_size=112,
        interpolation='bicubic',
        second_interpolation='lanczos',
        scale=(0.08, 1.0))
    module = build_from_cfg(transform, PIPELINES)
    fake_input = torch.rand((224, 224, 3)).numpy().astype(np.uint8)
    fake_input = Image.fromarray(fake_input)
    fake_output = module(fake_input)
    assert list(fake_output[0].size) == [224, 224]
    assert list(fake_output[1].size) == [112, 112]


def test_maskfeat_mask_gen():
    transform = dict(
        type='MaskFeatMaskGenerator', mask_window_size=14, mask_ratio=0.6)

    img = torch.rand((3, 224, 224))
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)
    assert list(res[1].shape) == [14, 14]
