# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

import pytest
import torch
from mmselfsup.models.algorithms import CAE

from mmengine.data import InstanceData

from mmselfsup.core.data_structures.selfsup_data_sample import \
    SelfSupDataSample
from mmselfsup.models.algorithms.cae import CAE

# model settings
backbone = dict(type='CAEViT', arch='b', patch_size=16, init_values=0.1)
neck = dict(
    type='CAENeck',
    patch_size=16,
    embed_dims=768,
    num_heads=12,
    regressor_depth=4,
    decoder_depth=4,
    mlp_ratio=4,
    init_values=0.1,
)
head = dict(type='CAEHead', tokenizer_path='cae_ckpt/encoder_stat_dict.pth')

loss = dict(type='CAELoss', lambd=2)

preprocess_cfg = {
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
    'to_rgb': True
}


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_cae():
    with pytest.raises(AssertionError):
        model = CAE(
            backbone=None,
            neck=neck,
            head=head,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        model = CAE(
            backbone=backbone,
            neck=None,
            head=head,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        model = CAE(
            backbone=backbone,
            neck=neck,
            head=None,
            loss=loss,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))
    with pytest.raises(AssertionError):
        model = CAE(
            backbone=backbone,
            neck=neck,
            head=head,
            loss=None,
            preprocess_cfg=copy.deepcopy(preprocess_cfg))

    model = CAE(
        backbone=backbone,
        neck=neck,
        head=head,
        loss=loss,
        preprocess_cfg=copy.deepcopy(preprocess_cfg))
    # model.init_weights()

    fake_img = torch.rand((3, 224, 224))
    fake_target_img = torch.rand((3, 112, 112))
    fake_mask = torch.zeros((196)).bool()
    fake_mask[75:150] = 1
    fake_data_sample = SelfSupDataSample()
    fake_mask = InstanceData(value=fake_mask)
    fake_data_sample.mask = fake_mask

    fake_data = [{
        'inputs': [fake_img, fake_target_img],
        'data_sample': fake_data_sample
    }]

    fake_loss = model(fake_data, return_loss=True)

    # test forward_train
    assert isinstance(fake_loss['loss'].item(), float)

    # test extract_feat
    fake_inputs, fake_data_samples = model.preprocss_data(fake_data)
    fake_feat = model.extract_feat(fake_inputs, fake_data_samples)

    assert list(fake_feat.shape) == [1, 122, 768]
