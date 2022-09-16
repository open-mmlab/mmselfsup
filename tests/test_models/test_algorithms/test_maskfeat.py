# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.structures import InstanceData

from mmselfsup.models.algorithms.maskfeat import MaskFeat
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

backbone = dict(type='MaskFeatViT', arch='b', patch_size=16)
loss = dict(type='MaskFeatReconstructionLoss')
head = dict(
    type='MaskFeatPretrainHead',
    predictor=dict(
        type='LinearNeck',
        in_channels=768,
        out_channels=108,
        with_avg_pool=False),
    loss=loss)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mae():
    data_preprocessor = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'bgr_to_rgb': True
    }
    hog_para = {'nbins': 9, 'pool': 8, 'gaussian_window': 16}

    alg = MaskFeat(
        backbone=backbone,
        head=head,
        hog_para=hog_para,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    # test forward_train
    fake_data_sample = SelfSupDataSample()
    fake_mask = InstanceData(value=torch.rand((14, 14)).bool())
    fake_data_sample.mask = fake_mask
    fake_data = {
        'inputs': [torch.randn((2, 3, 224, 224))],
        'data_sample': [fake_data_sample for _ in range(2)]
    }

    fake_batch_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_outputs = alg(fake_batch_inputs, fake_data_samples, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

    fake_feats = alg.extract_feat(fake_batch_inputs, fake_data_samples)
    assert list(fake_feats.shape) == [2, 197, 768]
