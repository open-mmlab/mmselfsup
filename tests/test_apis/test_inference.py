# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
from typing import List, Optional

import pytest
import torch
import torch.nn as nn
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


class ExampleModel(BaseModel):

    def __init__(self, backbone=backbone):
        super(ExampleModel, self).__init__(backbone=backbone)
        self.layer = nn.Linear(1, 1)

    def extract_feat(self,
                     inputs: List[torch.Tensor],
                     data_samples: Optional[List[SelfSupDataSample]] = None,
                     **kwargs) -> SelfSupDataSample:
        out = self.layer(inputs[0])
        return out


@pytest.mark.skipif(platform.system() == 'Windows', reason='')
def test_inference_model():
    register_all_modules()

    # Specify the data settings
    cfg = Config.fromfile(
        'configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py'  # noqa: E501
    )
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
