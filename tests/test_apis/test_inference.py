# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform

import pytest
import torch
import torch.nn as nn
from mmcv import Config
from PIL import Image

from mmselfsup.apis import inference_model
from mmselfsup.models import BaseModel


class ExampleModel(BaseModel):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.test_cfg = None
        self.layer = nn.Linear(1, 1)
        self.neck = nn.Identity()

    def extract_feat(self, imgs):
        pass

    def forward_train(self, imgs, **kwargs):
        pass

    def forward_test(self, img, **kwargs):
        out = self.layer(img)
        return out


@pytest.mark.skipif(platform.system() == 'Windows', reason='')
def test_inference_model():
    # Specify the data settings
    cfg = Config.fromfile(
        'configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py'  # noqa: E501
    )

    # Build the algorithm
    model = ExampleModel()
    model.cfg = cfg

    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.cfg.data = dict(
        test=dict(pipeline=[
            dict(type='Resize', size=(1, 1)),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))

    data = Image.open(
        osp.join(osp.dirname(__file__), '..', 'data', 'color.jpg'))

    # inference model
    data, output = inference_model(model, data)
    assert data.size() == torch.Size([1, 3, 1, 1])
    assert output.size() == torch.Size([1, 3, 1, 1])
