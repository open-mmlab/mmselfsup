# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.models.target_generators import HOGGenerator


def test_hog_generator():
    hog_generator = HOGGenerator()

    fake_input = torch.randn((2, 3, 224, 224))
    fake_output = hog_generator(fake_input)
    assert list(fake_output.shape) == [2, 196, 108]

    fake_hog_out = hog_generator.out[0].unsqueeze(0)
    fake_hog_img = hog_generator.generate_hog_image(fake_hog_out)
    assert fake_hog_img.shape == (224, 224)

    with pytest.raises(AssertionError):
        fake_hog_img = hog_generator.generate_hog_image(hog_generator.out[0])
