# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random

import numpy as np
import pytest
import torch
import torchvision
from mmcv import imread
from mmcv.transforms import Compose
from mmengine.utils import digit_version
from PIL import Image

import mmselfsup.datasets.transforms.processing as mmselfsup_transforms
from mmselfsup.datasets.transforms import (
    BEiTMaskGenerator, ColorJitter, RandomGaussianBlur, RandomPatchWithLabels,
    RandomResizedCropAndInterpolationWithTwoPic, RandomSolarize,
    RotationWithLabels, SimMIMMaskGenerator)


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
    module = RotationWithLabels(**transform)
    image = torch.rand((224, 224, 3)).numpy().astype(np.uint8)
    results = {'img': image}
    results = module(results)

    # test transform
    assert len(results['img']) == 4
    assert list(results['img'][0].shape) == [224, 224, 3]
    assert list(results['rot_label'].shape) == [4]

    assert isinstance(str(module), str)


def test_random_patch():
    transform = dict()
    module = RandomPatchWithLabels(**transform)
    image = torch.rand((224, 224, 3)).numpy().astype(np.uint8)
    results = {'img': image}
    results = module(results)

    # test transform
    assert len(results['img']) == 9
    assert list(results['img'][0].shape) == [53, 53, 3]
    assert list(results['patch_label'].shape) == [1, 8]
    assert list(results['patch_box'].shape) == [1, 9, 4]
    assert list(results['unpatched_img'].shape) == [1, 224, 224, 3]

    assert isinstance(str(module), str)


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


def test_randomresizedcrop():
    ori_img = imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    ori_img_pil = Image.open(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'))

    seed = random.randint(0, 100)

    # test when scale is not of kind (min, max)
    with pytest.raises(ValueError):
        kwargs = dict(
            size=(200, 300), scale=(1.0, 0.08), ratio=(3. / 4., 4. / 3.))
        aug = []
        aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        composed_transform(results)['img']

    # test when ratio is not of kind (min, max)
    with pytest.raises(ValueError):
        kwargs = dict(
            size=(200, 300), scale=(0.08, 1.0), ratio=(4. / 3., 3. / 4.))
        aug = []
        aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        composed_transform(results)['img']

    # test crop size is int
    kwargs = dict(size=200, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)

    # test __repr__()
    print(composed_transform)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (200, 200, 3)
    assert np.array(baseline).shape == (200, 200, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size < image size
    kwargs = dict(size=(200, 300), scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (200, 300, 3)
    assert np.array(baseline).shape == (200, 300, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size > image size
    kwargs = dict(size=(600, 700), scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (600, 700, 3)
    assert np.array(baseline).shape == (600, 700, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test cropping the whole image
    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        scale=(1.0, 2.0),
        ratio=(1.0, 2.0))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    assert np.array(baseline).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test central crop when in_ratio < min(ratio)
    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        scale=(1.0, 2.0),
        ratio=(2., 3.))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    assert np.array(baseline).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test central crop when in_ratio > max(ratio)
    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        scale=(1.0, 2.0),
        ratio=(3. / 4., 1))
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    assert np.array(baseline).shape == (ori_img.shape[0], ori_img.shape[1], 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test different interpolation types
    for mode in ['nearest', 'bilinear', 'bicubic', 'area', 'lanczos']:
        kwargs = dict(
            size=(600, 700),
            scale=(0.08, 1.0),
            ratio=(3. / 4., 4. / 3.),
            interpolation=mode)
        aug = []
        aug.extend([mmselfsup_transforms.RandomResizedCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        img = composed_transform(results)['img']
        assert img.shape == (600, 700, 3)


def test_randomcrop():
    ori_img = imread(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
    ori_img_pil = Image.open(
        osp.join(osp.dirname(__file__), '../../data/color.jpg'))
    seed = random.randint(0, 100)

    # test crop size is int
    kwargs = dict(size=200, padding=0, pad_if_needed=True, fill=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    kwargs = dict(size=200, padding=0, pad_if_needed=True, pad_val=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)

    # test __repr__()
    print(composed_transform)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (200, 200, 3)
    assert np.array(baseline).shape == (200, 200, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size < image size
    kwargs = dict(size=(200, 300), padding=0, pad_if_needed=True, fill=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    kwargs = dict(size=(200, 300), padding=0, pad_if_needed=True, pad_val=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (200, 300, 3)
    assert np.array(baseline).shape == (200, 300, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size > image size
    kwargs = dict(size=(600, 700), padding=0, pad_if_needed=True, fill=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    kwargs = dict(size=(600, 700), padding=0, pad_if_needed=True, pad_val=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']
    assert np.array(img).shape == (600, 700, 3)
    assert np.array(baseline).shape == (600, 700, 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test crop size == image size
    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        padding=0,
        pad_if_needed=True,
        fill=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    baseline = composed_transform(ori_img_pil)

    kwargs = dict(
        size=(ori_img.shape[0], ori_img.shape[1]),
        padding=0,
        pad_if_needed=True,
        pad_val=0)
    random.seed(seed)
    np.random.seed(seed)
    aug = []
    aug.extend([mmselfsup_transforms.RandomCrop(**kwargs)])
    composed_transform = Compose(aug)
    results = dict()
    results['img'] = ori_img
    img = composed_transform(results)['img']

    assert np.array(img).shape == (img.shape[0], img.shape[1], 3)
    assert np.array(baseline).shape == (img.shape[0], img.shape[1], 3)
    nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
    nonzero_transform = len((img - np.array(baseline)[:, :, ::-1]).nonzero())
    assert nonzero == nonzero_transform

    # test different padding mode
    for mode in ['constant', 'edge', 'reflect', 'symmetric']:
        kwargs = dict(size=(500, 600), padding=0, pad_if_needed=True, fill=0)
        kwargs['padding_mode'] = mode
        random.seed(seed)
        np.random.seed(seed)
        aug = []
        aug.extend([torchvision.transforms.RandomCrop(**kwargs)])
        composed_transform = Compose(aug)
        baseline = composed_transform(ori_img_pil)

        kwargs = dict(
            size=(500, 600), padding=0, pad_if_needed=True, pad_val=0)
        random.seed(seed)
        np.random.seed(seed)
        aug = []
        aug.extend([mmselfsup_transforms.RandomCrop(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        img = composed_transform(results)['img']
        assert np.array(img).shape == (500, 600, 3)
        assert np.array(baseline).shape == (500, 600, 3)
        nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
        nonzero_transform = len(
            (img - np.array(baseline)[:, :, ::-1]).nonzero())
        assert nonzero == nonzero_transform


def test_randomrotation():
    ori_img = torch.rand((224, 224, 3)).numpy().astype(np.uint8)
    ori_img_pil = Image.fromarray(ori_img, mode='RGB')

    seed = random.randint(0, 100)

    # test when degrees is negative
    with pytest.raises(ValueError):
        kwargs = dict(degrees=(-30))
        aug = []
        aug.extend([mmselfsup_transforms.RandomRotation(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        composed_transform(results)['img']

    # test when center is not None and expand=True
    with pytest.raises(ValueError):
        kwargs = dict(degrees=(30, 60), expand=True, center=(5, 3))
        aug = []
        aug.extend([mmselfsup_transforms.RandomRotation(**kwargs)])
        composed_transform = Compose(aug)
        results = dict()
        results['img'] = ori_img
        composed_transform(results)['img']

    kwargs_list = [
        # test degrees
        dict(degrees=(30, 60)),
        dict(degrees=(30, 30)),
        dict(degrees=(60, 370)),
        # test different interpolation types
        dict(degrees=(30, 60), interpolation='nearest'),
        dict(degrees=(30, 60), interpolation='bilinear'),
        dict(degrees=(30, 60), interpolation='bicubic'),
        # test center
        dict(degrees=(30, 60), center=(5, 3)),
        # test fill
        dict(degrees=(30, 60), fill=5)
    ]

    for kwargs in kwargs_list:
        # RandomRotation in mmselfsup
        random.seed(seed)
        np.random.seed(seed)

        aug = []
        aug.extend([mmselfsup_transforms.RandomRotation(**kwargs)])
        composed_transform = Compose(aug)

        # test __repr__()
        print(composed_transform)

        results = dict()
        results['img'] = ori_img
        img = composed_transform(results)['img']

        # RandomRotation in torchvision
        random.seed(seed)
        np.random.seed(seed)

        if 'interpolation' in kwargs:
            if digit_version(
                    torchvision.__version__) >= digit_version('0.9.0'):
                from torchvision.transforms.functional import InterpolationMode
                inverse_modes_mapping = {
                    'nearest': InterpolationMode.NEAREST,
                    'bilinear': InterpolationMode.BILINEAR,
                    'bicubic': InterpolationMode.BICUBIC,
                }

                mode = kwargs['interpolation']
                kwargs['interpolation'] = inverse_modes_mapping[mode]
            else:
                kwargs.pop('interpolation')

        aug = []
        aug.extend([torchvision.transforms.RandomRotation(**kwargs)])
        composed_transform = Compose(aug)
        baseline = composed_transform(ori_img_pil)

        # compare the outputs of RandomRotation in mmselfsup and torchvision
        assert np.array(img).shape == np.array(baseline).shape
        nonzero = len((ori_img - np.array(ori_img_pil)[:, :, ::-1]).nonzero())
        nonzero_transform = len(
            (img - np.array(baseline)[:, :, ::-1]).nonzero())
        assert nonzero == nonzero_transform
