# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine import Runner
from mmengine.data import LabelData
from mmengine.model import BaseModel as EngineBaseModel
from mmengine.model import BaseModule
from mmengine.optim import OptimWrapper
from torch.utils.data import Dataset

from mmselfsup.engine import DenseCLHook
from mmselfsup.models.algorithms import BaseModel
from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        data_sample = SelfSupDataSample()
        gt_label = LabelData(value=self.label[index])
        setattr(data_sample, 'gt_label', gt_label)
        return dict(inputs=self.data[index], data_sample=data_sample)


@MODELS.register_module()
class DenseCLDummyLayer(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__(backbone=dict(type='DenseCLDummyLayer'))
        self.loss_lambda = 0.5

    def loss(self, batch_inputs, data_samples):
        labels = []
        for x in data_samples:
            labels.append(x.gt_label.value)
            labels = torch.stack(labels)
        outputs = self.backbone(batch_inputs)
        loss = (labels - outputs).sum()
        outputs = dict(loss=loss)
        return outputs


class TestDenseCLHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_densecl_hook(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        dummy_dataset = DummyDataset()
        toy_model = ToyModel().to(device)
        densecl_hook = DenseCLHook(start_iters=1)

        class DummyWrapper(EngineBaseModel):

            def __init__(self, model):
                super().__init__()
                self.module = model

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        # test DenseCLHook with model wrapper
        runner = Runner(
            model=DummyWrapper(toy_model),
            work_dir=self.temp_dir.name,
            train_dataloader=dict(
                dataset=dummy_dataset,
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=0),
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(toy_model.parameters())),
            param_scheduler=dict(type='MultiStepLR', milestones=[1]),
            train_cfg=dict(by_epoch=True, max_epochs=2),
            custom_hooks=[densecl_hook],
            default_hooks=dict(logger=None),
            log_processor=dict(window_size=1),
            experiment_name='test_densecl_hook')

        runner.train()

        if runner.iter >= 1:
            assert runner.model.module.loss_lambda == 0.5
        else:
            assert runner.model.module.loss_lambda == 0.
