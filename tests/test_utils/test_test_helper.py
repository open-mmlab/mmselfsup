# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mmselfsup.utils.test_helper import single_gpu_test


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([1]), img_metas=dict())
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, img, mode='test', **kwargs):
        return dict(img=img)

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


def test_test_helper():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    model = ExampleModel()

    res = single_gpu_test(model, data_loader)
    assert res['img'] == np.array([[1]])
