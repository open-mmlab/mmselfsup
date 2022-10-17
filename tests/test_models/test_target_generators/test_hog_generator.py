# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.target_generators import HOGGenerator


def test_hog_generator():
    hog_generator = HOGGenerator()

    fake_input = torch.randn((2, 3, 224, 224))
    fake_output = hog_generator(fake_input)
    assert list(fake_output.shape) == [2, 196, 108]
