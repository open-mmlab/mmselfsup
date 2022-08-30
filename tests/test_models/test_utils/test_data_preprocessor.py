# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.utils import SelfSupDataPreprocessor
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
