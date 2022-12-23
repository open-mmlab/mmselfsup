# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.target_generators import Encoder


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_dalle():
    model = Encoder()
    fake_inputs = torch.rand((2, 3, 112, 112))
    fake_outputs = model(fake_inputs)

    assert list(fake_outputs.shape) == [2, 8192, 14, 14]
