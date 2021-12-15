# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
import torch
from mmcv.utils import build_from_cfg
from PIL import Image

from mmselfsup.datasets.builder import PIPELINES


def test_random_applied_trans():
    img = Image.open(osp.join(osp.dirname(__file__), '../data/color.jpg'))

    # p=0.5
    transform = dict(
        type='RandomAppliedTrans', transforms=[dict(type='Solarization')])
    module = build_from_cfg(transform, PIPELINES)
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
    img = np.array(
        Image.open(osp.join(osp.dirname(__file__), '../data/color.jpg')))
    with pytest.raises(AssertionError):
        res = module(img)

    img = torch.from_numpy(img).float().permute(2, 1, 0)
    res = module(img)

    assert img.size() == res.size()


def test_gaussianblur():
    with pytest.raises(AssertionError):
        transform = dict(
            type='GaussianBlur', sigma_min=0.1, sigma_max=1.0, p=-1)
        module = build_from_cfg(transform, PIPELINES)

    img = Image.open(osp.join(osp.dirname(__file__), '../data/color.jpg'))

    # p=0.5
    transform = dict(type='GaussianBlur', sigma_min=0.1, sigma_max=1.0)
    module = build_from_cfg(transform, PIPELINES)
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

    img = Image.open(osp.join(osp.dirname(__file__), '../data/color.jpg'))

    # p=0.5
    transform = dict(type='Solarization')
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)

    transform = dict(type='Solarization', p=0.)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)

    transform = dict(type='Solarization', p=1.)
    module = build_from_cfg(transform, PIPELINES)
    res = module(img)

    assert img.size == res.size
