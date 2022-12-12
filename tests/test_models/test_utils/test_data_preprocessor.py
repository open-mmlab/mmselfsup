# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.models.utils import (SelfSupDataPreprocessor,
                                    TwoNormDataPreprocessor)
from mmselfsup.structures import SelfSupDataSample


def test_selfsup_data_preprocessor():
    data_preprocessor = SelfSupDataPreprocessor(
        rgb_to_bgr=True, mean=[124, 117, 104], std=[59, 58, 58])
    fake_data = {
        'inputs': [torch.randn((2, 3, 224, 224))],
        'data_sample': [SelfSupDataSample(),
                        SelfSupDataSample()]
    }
    fake_batches, fake_samples = data_preprocessor(fake_data)
    assert len(fake_batches) == 1
    assert len(fake_samples) == 2


def test_two_norm_data_preprocessor():
    with pytest.raises(AssertionError):
        data_preprocessor = TwoNormDataPreprocessor(
            rgb_to_bgr=True,
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        )
    with pytest.raises(AssertionError):
        data_preprocessor = TwoNormDataPreprocessor(
            rgb_to_bgr=True,
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            second_mean=(127.5, 127.5),
            second_std=(127.5, 127.5, 127.5),
        )
    with pytest.raises(AssertionError):
        data_preprocessor = TwoNormDataPreprocessor(
            rgb_to_bgr=True,
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
            second_mean=(127.5, 127.5, 127.5),
            second_std=(127.5, 127.5),
        )

    data_preprocessor = dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        second_mean=(127.5, 127.5, 127.5),
        second_std=(127.5, 127.5, 127.5),
        bgr_to_rgb=True)

    data_preprocessor = TwoNormDataPreprocessor(**data_preprocessor)
    fake_data = {
        'inputs':
        [torch.randn((4, 3, 224, 224)),
         torch.randn((4, 3, 224, 224))],
        'data_sample': [
            SelfSupDataSample(),
            SelfSupDataSample(),
            SelfSupDataSample(),
            SelfSupDataSample()
        ]
    }
    fake_batches, fake_samples = data_preprocessor(fake_data)
    assert len(fake_batches) == 2
    assert len(fake_samples) == 4
