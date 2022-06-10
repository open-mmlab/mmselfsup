# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.core import SelfSupDataSample
from mmselfsup.models.utils import SelfSupDataPreprocessor


def test_selfsup_data_preprocessor():
    data_preprocessor = SelfSupDataPreprocessor(rgb_to_bgr=True)
    fake_data = [{
        'inputs': [torch.randn((3, 224, 224))],
        'data_sample': SelfSupDataSample()
    } for _ in range(2)]
    fake_batches, fake_samples = data_preprocessor(fake_data)
    assert len(fake_batches) == 1
    assert len(fake_samples) == 2
