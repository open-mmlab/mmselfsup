# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import MaskFeat

backbone = dict(type='MaskFeatViT', arch='b', patch_size=16, mask_ratio=0.4)
head = dict(type='MaskFeatPretrainHead')


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mae():
    with pytest.raises(AssertionError):
        alg = MaskFeat(backbone=backbone, head=None)
    with pytest.raises(AssertionError):
        alg = MaskFeat(backbone=None, head=head)
    alg = MaskFeat(backbone=backbone, head=head)

    fake_input = torch.randn((2, 3, 224, 224))
    fake_hog = torch.randn((2, 108, 14, 14))
    fake_loss = alg.forward_train(fake_input, fake_hog)
    fake_feature = alg.extract_feat(fake_input)[0]
    assert isinstance(fake_loss['loss'].item(), float)
    assert list(fake_feature.shape) == [2, 196, 768]
