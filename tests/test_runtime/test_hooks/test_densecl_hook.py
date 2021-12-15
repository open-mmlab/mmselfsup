# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_runner, obj_from_dict
from torch.utils.data import DataLoader, Dataset

from mmselfsup.core.hooks import DenseCLHook


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
        self.loss_lambda = 0.5
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


def test_densecl_hook():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)

    runner_cfg = dict(type='EpochBasedRunner', max_epochs=2)
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

    # test DenseCLHook
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MMDataParallel(ExampleModel())
        optimizer = obj_from_dict(optim_cfg, torch.optim,
                                  dict(params=model.parameters()))

        densecl_hook = DenseCLHook(start_iters=1)
        runner = build_runner(
            runner_cfg,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logging.getLogger()))
        runner.register_hook(densecl_hook)
        runner.run([data_loader], [('train', 1)])
        cur_iter = runner.iter
        if cur_iter >= 1:
            assert runner.model.module.loss_lambda == 0.5
        else:
            assert runner.model.module.loss_lambda == 0.
