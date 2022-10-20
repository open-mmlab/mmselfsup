# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.evaluation import knn_eval


def test_knn_eval():
    train_feats = torch.ones(200, 3)
    train_labels = torch.ones(200).long()
    test_feats = torch.ones(200, 3)
    test_labels = torch.ones(200).long()
    num_knn = [10, 20, 100, 200]
    for k in num_knn:
        top1, top5 = knn_eval(train_feats, train_labels, test_feats,
                              test_labels, k, 0.07)
        assert top1 == 100.
        assert top5 == 100.
