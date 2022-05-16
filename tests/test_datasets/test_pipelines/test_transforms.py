# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmselfsup.datasets.pipelines import (
    BEiTMaskGenerator, ColorJitter, Lighting, RandomGaussianBlur,
    RandomPatchWithLabels, RandomResizedCropAndInterpolationWithTwoPic,
    RandomRotationWithLabels, RandomSolarize, SimMIMMaskGenerator)


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

    results = {'img': fake_input}
    results = module(results)
    # test transform
    assert list(results['img'][0].shape) == [224, 224, 3]
    assert list(results['img'][1].shape) == [112, 112, 3]

    # test repr
    assert isinstance(str(module), str)


def test_lighting():
    with pytest.raises(AssertionError):
        transform = Lighting(eigval=1)

    with pytest.raises(AssertionError):
        transform = Lighting(eigvec=1)

    with pytest.raises(AssertionError):
        transform = Lighting(eigvec=[1])

    original_img = np.ones((8, 8, 3), dtype=np.uint8)
    results = dict(img=original_img)

    transform = Lighting()
    assert isinstance(str(transform), str)

    results = transform(results)
    assert results['img'].shape == original_img.shape

    transform = Lighting(alphastd=0., to_rgb=False)
    results = transform(dict(img=original_img))
    assert np.equal(results['img'], original_img).all()


def test_random_gaussiablur():
    with pytest.raises(AssertionError):
        transform = RandomGaussianBlur(sigma_min=0.1, sigma_max=1.0, prob=-1)

    original_img = np.ones((8, 8, 3), dtype=np.uint8)
    results = dict(img=original_img)

    transform = RandomGaussianBlur(sigma_min=0.1, sigma_max=1.0)
    assert isinstance(str(transform), str)

    results = transform(results)
    assert results['img'].shape == original_img.shape


def test_random_solarize():
    with pytest.raises(AssertionError):
        transform = RandomSolarize(prob=-1)

    original_img = np.ones((8, 8, 3), dtype=np.uint8)
    results = dict(img=original_img)

    transform = RandomSolarize()
    assert isinstance(str(transform), str)

    results = transform(results)
    assert results['img'].shape == original_img.shape


def test_random_rotation():
    transform = dict()
    module = RandomRotationWithLabels(**transform)
    image = torch.rand((224, 224, 3)).numpy().astype(np.uint8)
    results = {'img': image}
    results = module(results)

    # test transform
    assert list(results['img'].shape) == [4, 3, 224, 224]
    assert list(results['rot_label'].shape) == [4]


def test_random_patch():
    transform = dict()
    module = RandomPatchWithLabels(**transform)
    image = torch.rand((224, 224, 3)).numpy().astype(np.uint8)
    results = {'img': image}
    results = module(results)

    # test transform
    assert list(results['img'].shape) == [8, 6, 53, 53]
    assert list(results['patch_label'].shape) == [8]


def test_color_jitter():
    with pytest.raises(ValueError):
        transform = ColorJitter(-1, 0, 0, 0)

    with pytest.raises(ValueError):
        transform = ColorJitter(0, 0, 0, [0, 1])

    with pytest.raises(TypeError):
        transform = ColorJitter('test', 0, 0, 0)

    original_img = torch.rand((224, 224, 3)).numpy().astype(np.uint8)
    results = {'img': original_img}

    transform = ColorJitter(0, 0, 0, 0)
    results = transform(results)
    assert np.equal(results['img'], original_img).all()

    transform = ColorJitter(0.4, 0.4, 0.2, 0.1)
    results = transform(results)
    assert results['img'].shape == original_img.shape

    assert isinstance(str(transform), str)
