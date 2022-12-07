# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmengine.structures import InstanceData
from mmengine.utils import digit_version

from mmselfsup.models.algorithms.maskfeat import MaskFeat
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

register_all_modules()

backbone = dict(type='MaskFeatViT', arch='b', patch_size=16)
neck = dict(
    type='LinearNeck', in_channels=768, out_channels=108, with_avg_pool=False)
head = dict(
    type='MaskFeatPretrainHead',
    loss=dict(type='PixelReconstructionLoss', criterion='L2'))
target_generator = dict(
    type='HOGGenerator', nbins=9, pool=8, gaussian_window=16)


@pytest.mark.skipif(
    digit_version(torch.__version__) < digit_version('1.7.0'),
    reason='torch version')
@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_maskfeat():
    data_preprocessor = {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'bgr_to_rgb': True
    }

    alg = MaskFeat(
        backbone=backbone,
        neck=neck,
        head=head,
        target_generator=target_generator,
        data_preprocessor=copy.deepcopy(data_preprocessor))

    # test forward_train
    fake_data_sample = SelfSupDataSample()
    fake_mask = InstanceData(value=torch.rand((14, 14)).bool())
    fake_data_sample.mask = fake_mask
    fake_data = {
        'inputs': [torch.randn((1, 3, 224, 224))],
        'data_sample': [fake_data_sample for _ in range(1)]
    }

    fake_batch_inputs, fake_data_samples = alg.data_preprocessor(fake_data)
    fake_outputs = alg(fake_batch_inputs, fake_data_samples, mode='loss')
    assert isinstance(fake_outputs['loss'].item(), float)

    # test extraction
    fake_feats = alg.extract_feat(fake_batch_inputs, fake_data_samples)
    assert list(fake_feats.shape) == [1, 196, 108]

    # test reconstruction
    results = alg.reconstruct(fake_feats, fake_data_samples)
    assert list(results.mask.value.shape) == [1, 224, 224, 3]
    assert list(results.pred.value.shape) == [1, 224, 224, 3]
