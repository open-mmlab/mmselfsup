# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel
from torch.utils.data import DataLoader, Dataset

from mmselfsup.models.utils import ExtractProcess, MultiExtractProcess


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([1]), img_metas=dict())
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, img, test_mode=False, **kwargs):
        return [
            torch.rand((1, 32, 112, 112)),
            torch.rand((1, 64, 56, 56)),
            torch.rand((1, 128, 28, 28)),
        ]

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


def test_extract_process():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    model = MMDataParallel(ExampleModel())

    process = ExtractProcess()

    results = process.extract(model, data_loader)
    assert 'feat' in results
    assert results['feat'].shape == (1, 128 * 1 * 1)


def test_multi_extract_process():
    with pytest.raises(AssertionError):
        process = MultiExtractProcess(
            pool_type='specified', backbone='resnet50', layer_indices=(-1, ))

    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    model = MMDataParallel(ExampleModel())

    process = MultiExtractProcess(
        pool_type='specified', backbone='resnet50', layer_indices=(0, 1, 2))

    results = process.extract(model, data_loader)
    assert 'feat1' in results
    assert 'feat2' in results
    assert 'feat3' in results
    assert results['feat1'].shape == (1, 32 * 12 * 12)
    assert results['feat2'].shape == (1, 64 * 6 * 6)
    assert results['feat3'].shape == (1, 128 * 4 * 4)
