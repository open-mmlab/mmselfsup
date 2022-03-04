# Copyright (c) OpenMMLab. All rights reserved.
import logging
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_runner, obj_from_dict
from torch.utils.data import DataLoader, Dataset

from mmselfsup.core.hooks import DistOptimizerHook, GradAccumFp16OptimizerHook


class ExampleDataset(Dataset):

    def __getitem__(self, idx):
        results = dict(img=torch.tensor([1.]), img_metas=dict())
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.linear = nn.Linear(1, 1)
        self.prototypes_test = nn.Linear(1, 1)

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        out = self.linear(img)
        out = self.prototypes_test(out)
        return out

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss, num_samples=len(data_batch))


def test_optimizer_hook():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)

    runner_cfg = dict(type='EpochBasedRunner', max_epochs=5)
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optim_hook_cfg = dict(
        grad_clip=dict(max_norm=10), frozen_layers_cfg=dict(prototypes=5005))

    optimizer_hook = DistOptimizerHook(**optim_hook_cfg)

    # test DistOptimizerHook
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MMDataParallel(ExampleModel())
        optimizer = obj_from_dict(optim_cfg, torch.optim,
                                  dict(params=model.parameters()))

        runner = build_runner(
            runner_cfg,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logging.getLogger()))
        runner.register_training_hooks(optimizer_hook)

        prototypes_start = []
        for name, p in runner.model.module.named_parameters():
            if 'prototypes_test' in name:
                prototypes_start.append(p)

        # run training
        runner.run([data_loader], [('train', 1)])

        prototypes_end = []
        for name, p in runner.model.module.named_parameters():
            if 'prototypes_test' in name:
                prototypes_end.append(p)

        assert len(prototypes_start) == len(prototypes_end)
        for i in range(len(prototypes_start)):
            p_start = prototypes_start[i]
            p_end = prototypes_end[i]
            assert p_start == p_end


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is not available.')
def test_fp16optimizer_hook():
    test_dataset = ExampleDataset()
    test_dataset.evaluate = MagicMock(return_value=dict(test='success'))
    data_loader = DataLoader(
        test_dataset, batch_size=1, sampler=None, num_workers=0, shuffle=False)

    runner_cfg = dict(type='EpochBasedRunner', max_epochs=5)
    optim_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
    optim_hook_cfg = dict(
        grad_clip=dict(max_norm=10),
        loss_scale=16.,
        frozen_layers_cfg=dict(prototypes=5005))

    optimizer_hook = GradAccumFp16OptimizerHook(**optim_hook_cfg)

    # test GradAccumFp16OptimizerHook
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MMDataParallel(ExampleModel())
        optimizer = obj_from_dict(optim_cfg, torch.optim,
                                  dict(params=model.parameters()))

        runner = build_runner(
            runner_cfg,
            default_args=dict(
                model=model,
                optimizer=optimizer,
                work_dir=tmpdir,
                logger=logging.getLogger(),
                meta=dict()))
        runner.register_training_hooks(optimizer_hook)
        # run training
        runner.run([data_loader], [('train', 1)])
        assert runner.meta['fp16']['loss_scaler']['scale'] == 16.
