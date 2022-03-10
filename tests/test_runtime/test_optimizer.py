# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmselfsup.core import LARS, build_optimizer


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.predictor = nn.Linear(2, 1)

    def forward(self, img, img_metas, test_mode=False, **kwargs):
        res = self.predictor(img)
        return res

    def train_step(self, data_batch, optimizer):
        loss = self.forward(**data_batch)
        return dict(loss=loss.sum())


def test_lars():
    optimizer = dict(
        type='LARS',
        lr=0.3,
        momentum=0.9,
        weight_decay=1e-6,
        paramwise_options={'bias': dict(weight_decay=0., lars_exclude=True)})

    model = ExampleModel()
    optimizer = build_optimizer(model, optimizer)

    for i in range(2):
        loss = model.train_step(
            dict(img=torch.ones(2, 2), img_metas=None), optimizer)

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

    with pytest.raises(ValueError):
        optimizer = LARS(model.parameters(), lr=-1)
    with pytest.raises(ValueError):
        optimizer = LARS(model.parameters(), lr=0.1, momentum=-1)
    with pytest.raises(ValueError):
        optimizer = LARS(model.parameters(), lr=0.1, weight_decay=-1)
    with pytest.raises(ValueError):
        optimizer = LARS(model.parameters(), lr=0.1, eta=-1)
