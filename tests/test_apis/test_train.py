# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import tempfile
import time

import mmcv
import pytest
import torch
import torch.nn as nn
from mmcv import Config
from torch.utils.data import Dataset

from mmselfsup.apis import init_random_seed, set_random_seed, train_model


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(
            img=torch.tensor([1], dtype=torch.float32), img_metas=dict())
        return results

    def __len__(self):
        return 2


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.layer = nn.Linear(1, 1)
        self.neck = nn.Identity()

    def forward(self, img, test_mode=False, **kwargs):
        out = self.layer(img)
        return out

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


@pytest.mark.skipif(platform.system() == 'Windows', reason='')
def test_train_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Specify the data settings
        cfg = Config.fromfile(
            'configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py'  # noqa: E501
        )

        cfg.data.samples_per_gpu = 1
        cfg.data.workers_per_gpu = 2

        cfg.data.val.data_source.data_prefix = 'tests/data/'
        cfg.data.val.data_source.ann_file = 'tests/data/data_list.txt'

        # Specify the optimizer
        cfg.optimizer = dict(
            type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
        cfg.optimizer_config = dict(grad_clip=None)

        # Specify the learning rate scheduler
        cfg.lr_config = dict(policy='step', step=[1])

        # Modify runtime setting
        cfg.runner = dict(type='EpochBasedRunner', max_epochs=1)

        # Specify the work directory
        cfg.work_dir = tmpdir

        # Set the random seed and enable the deterministic option of cuDNN
        # to keep the results' reproducible
        cfg.seed = 0
        set_random_seed(0, deterministic=True)

        cfg.gpu_ids = range(1)

        # Create the work directory
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Build the algorithm
        model = ExampleModel()

        # Build the dataset
        datasets = [ExampleDataset()]

        # evaluation
        cfg.evaluation = dict(interval=10, topk=(1, 5))

        # Start pre-train
        train_model(
            model,
            datasets,
            cfg,
            distributed=False,
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            meta=dict())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is not available')
def test_init_random_seed():
    seed = init_random_seed(0)
    assert seed == 0


def test_set_random_seed():
    set_random_seed(0)
