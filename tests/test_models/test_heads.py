# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmselfsup.models.heads import (ClsHead, ContrastiveHead, LatentClsHead,
                                    LatentPredictHead, MAEFinetuneHead,
                                    MAEPretrainHead, MultiClsHead, SwAVHead)


def test_cls_head():
    # test ClsHead
    head = ClsHead()
    fake_cls_score = [torch.rand(4, 3)]
    fake_gt_label = torch.randint(0, 2, (4, ))

    loss = head.loss(fake_cls_score, fake_gt_label)
    assert loss['loss'].item() > 0


def test_contrastive_head():
    head = ContrastiveHead()
    fake_pos = torch.rand(32, 1)  # N, 1
    fake_neg = torch.rand(32, 100)  # N, k

    loss = head.forward(fake_pos, fake_neg)
    assert loss['loss'].item() > 0


def test_latent_predict_head():
    predictor = dict(
        type='NonLinearNeck',
        in_channels=64,
        hid_channels=128,
        out_channels=64,
        with_bias=True,
        with_last_bn=True,
        with_avg_pool=False,
        norm_cfg=dict(type='BN1d'))
    head = LatentPredictHead(predictor=predictor)
    fake_input = torch.rand(32, 64)  # N, C
    fake_traget = torch.rand(32, 64)  # N, C

    loss = head.forward(fake_input, fake_traget)
    assert loss['loss'].item() > -1


def test_latent_cls_head():
    head = LatentClsHead(64, 10)
    fake_input = torch.rand(32, 64)  # N, C
    fake_traget = torch.rand(32, 64)  # N, C

    loss = head.forward(fake_input, fake_traget)
    assert loss['loss'].item() > 0


def test_multi_cls_head():
    head = MultiClsHead(in_indices=(0, 1))
    fake_input = [torch.rand(8, 64, 5, 5), torch.rand(8, 256, 14, 14)]
    out = head.forward(fake_input)
    assert isinstance(out, list)

    fake_cls_score = [torch.rand(4, 3)]
    fake_gt_label = torch.randint(0, 2, (4, ))

    loss = head.loss(fake_cls_score, fake_gt_label)
    print(loss.keys())
    for k in loss.keys():
        if 'loss' in k:
            assert loss[k].item() > 0


def test_swav_head():
    head = SwAVHead(feat_dim=128, num_crops=[2, 6])
    fake_input = torch.rand(32, 128)  # N, C

    loss = head.forward(fake_input)
    assert loss['loss'].item() > 0


def test_mae_pretrain_head():
    head = MAEPretrainHead(norm_pix=False, patch_size=16)
    fake_input = torch.rand((2, 3, 224, 224))
    fake_mask = torch.ones((2, 196))
    fake_pred = torch.rand((2, 196, 768))

    loss = head.forward(fake_input, fake_pred, fake_mask)

    assert loss['loss'].item() > 0

    head_norm_pixel = MAEPretrainHead(norm_pix=True, patch_size=16)

    loss_norm_pixel = head_norm_pixel.forward(fake_input, fake_pred, fake_mask)

    assert loss_norm_pixel['loss'].item() > 0


def test_mae_finetune_head():

    head = MAEFinetuneHead(num_classes=1000, embed_dim=768)
    fake_input = torch.rand((2, 768))
    fake_labels = F.normalize(torch.rand((2, 1000)), dim=-1)
    fake_features = head.forward(fake_input)

    assert list(fake_features[0].shape) == [2, 1000]

    loss = head.loss(fake_features, fake_labels)

    assert loss['loss'].item() > 0
