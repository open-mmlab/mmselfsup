# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
<<<<<<< HEAD
from typing import List, Optional
=======
>>>>>>> upstream/master

import pytest
import torch
import torch.nn as nn
<<<<<<< HEAD
from mmengine.config import Config

from mmselfsup.apis import inference_model
from mmselfsup.models import BaseModel
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import register_all_modules

backbone = dict(
    type='ResNet',
    depth=18,
    in_channels=2,
    out_indices=[4],  # 0: conv-1, x: stage-x
    norm_cfg=dict(type='BN'))
=======
from mmcv import Config
from PIL import Image

from mmselfsup.apis import inference_model
from mmselfsup.models import BaseModel
>>>>>>> upstream/master


class ExampleModel(BaseModel):

<<<<<<< HEAD
    def __init__(self, backbone=backbone):
        super(ExampleModel, self).__init__(backbone=backbone)
        self.layer = nn.Linear(1, 1)

    def extract_feat(self,
                     inputs: List[torch.Tensor],
                     data_samples: Optional[List[SelfSupDataSample]] = None,
                     **kwargs) -> SelfSupDataSample:
        out = self.layer(inputs[0])
=======
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
>>>>>>> upstream/master
        return out


@pytest.mark.skipif(platform.system() == 'Windows', reason='')
def test_inference_model():
<<<<<<< HEAD
    register_all_modules()

=======
>>>>>>> upstream/master
    # Specify the data settings
    cfg = Config.fromfile(
        'configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py'  # noqa: E501
    )
<<<<<<< HEAD
    # Build the algorithm
    model = ExampleModel()
    model.cfg = cfg
    model.cfg.test_dataloader = dict(
        dataset=dict(pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(1, 1)),
            dict(type='PackSelfSupInputs', meta_keys=['img_path'])
        ]))

    img_path = osp.join(osp.dirname(__file__), '..', 'data', 'color.jpg')

    # inference model
    out = inference_model(model, img_path)
    assert out.size() == torch.Size([1, 3, 1, 1])
=======

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
>>>>>>> upstream/master
