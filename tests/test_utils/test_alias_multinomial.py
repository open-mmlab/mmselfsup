# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmselfsup.utils import AliasMethod


def test_alias_multinomial():
    example_in = torch.Tensor([1, 2, 3, 4])
    example_alias_method = AliasMethod(example_in)
    assert (example_alias_method.prob.numpy() <= 1).all()
    assert len(example_in) == len(example_alias_method.alias)

    # test assertion if N is smaller than 0
    with pytest.raises(AssertionError):
        example_alias_method.draw(-1)
    with pytest.raises(AssertionError):
        example_alias_method.draw(0)

    example_res = example_alias_method.draw(5)
    assert len(example_res) == 5
