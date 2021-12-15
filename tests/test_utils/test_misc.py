# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.utils.misc import tensor2imgs


def test_tensor2imgs():
    with pytest.raises(AssertionError):
        tensor2imgs(torch.rand((3, 16, 16)))
    fake_tensor = torch.rand((3, 3, 16, 16))
    fake_imgs = tensor2imgs(fake_tensor)
    assert len(fake_imgs) == 3
    assert fake_imgs[0].shape == (16, 16, 3)
