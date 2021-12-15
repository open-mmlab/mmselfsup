# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_runner
from torch.utils.data import DataLoader, Dataset

from mmselfsup.core.hooks import SimSiamHook
from mmselfsup.core.optimizer import build_optimizer


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
        self.predictor = nn.Linear(2, 1)

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        return img

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss)


def test_simsiam_hook():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))

    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)
    runner_cfg = dict(type='EpochBasedRunner', max_epochs=2)
    optim_cfg = dict(
        type='SGD',
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0005,
        paramwise_options={'predictor': dict(fix_lr=True)})
    lr_config = dict(policy='CosineAnnealing', min_lr=0.)

    # test SimSiamHook
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MMDataParallel(ExampleModel())
        optimizer = build_optimizer(model, optim_cfg)
        simsiam_hook = SimSiamHook(True, 0.05)
        runner = build_runner(
            runner_cfg,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logging.getLogger()))
        runner.register_training_hooks(lr_config)
        runner.register_hook(simsiam_hook)
        runner.run([data_loader], [('train', 1)])

        for param_group in runner.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                assert param_group['lr'] == 0.05
            else:
                assert param_group['lr'] != 0.05
