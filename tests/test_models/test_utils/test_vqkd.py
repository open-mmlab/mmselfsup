# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.utils import VQKD


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_dalle():
    model = VQKD(img_size=224)
    fake_inputs = torch.rand((2, 3, 224, 224))
    fake_outputs = model(fake_inputs)

    assert list(fake_outputs.shape) == [2, 196]
