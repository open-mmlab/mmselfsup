# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_runner
from torch.utils.data import Dataset

from mmselfsup.core.optimizer import build_optimizer
from mmselfsup.utils import Extractor


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
        self.neck = nn.Identity()

    def forward(self, img, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


def test_extractor():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))

    runner_cfg = dict(type='EpochBasedRunner', max_epochs=2)
    optim_cfg = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
    extractor = Extractor(
        test_dataset, 1, 0, dist_mode=False, persistent_workers=False)

    # test extractor
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MMDataParallel(ExampleModel())
        optimizer = build_optimizer(model, optim_cfg)
        runner = build_runner(
            runner_cfg,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logging.getLogger()))
        features = extractor(runner)
        assert features.shape == (1, 1)
