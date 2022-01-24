# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.algorithms import MAEClassification

model_finetune = dict(
    backbone=dict(
        type='MAEClsViT',
        arch='b',
        patch_size=16,
        global_pool=True,
        drop_path_rate=0.1,
        final_norm=False),
    head=dict(type='MAEFinetuneHead', num_classes=1000, embed_dim=768),
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=1.0,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=1000,
    finetune=True)

model_linprobe = dict(
    backbone=dict(
        type='MAEClsViT',
        arch='b',
        patch_size=16,
        global_pool=True,
        finetune=False,
        final_norm=False),
    head=dict(type='MAELinprobeHead', num_classes=1000, embed_dim=768),
    finetune=False)


def test_mae():

    alg_ft = MAEClassification(**model_finetune).cpu()
    alg_linprobe = MAEClassification(**model_linprobe)

    fake_input = torch.randn((16, 3, 224, 224))
    fake_target = torch.ones((16, )).long()
    fake_ft_loss = alg_ft.forward_train(fake_input, fake_target)
    fake_ft_feature = alg_ft.forward_test(fake_input)

    fake_linprobe_loss = alg_linprobe.forward_train(fake_input, fake_target)
    fake_linprobe_feature = alg_linprobe.forward_test(fake_input)

    assert isinstance(fake_ft_loss['loss'].item(), float)
    assert isinstance(fake_linprobe_loss['loss'].item(), float)
    assert list(fake_ft_feature['last_layer'].shape) == [16, 1000]
    assert list(fake_linprobe_feature['last_layer'].shape) == [16, 1000]
