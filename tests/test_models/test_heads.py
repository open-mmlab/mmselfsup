# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.heads import (ClsHead, ContrastiveHead, LatentClsHead,
                                    LatentPredictHead, MultiClsHead, SwAVHead)


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
