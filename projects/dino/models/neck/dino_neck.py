# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class DINONeck(BaseModule):

    def __init__(self) -> None:
        super().__init__()
        # TODO: implement the initialization function

    def forward(self):
        """Forward function of DINO Neck."""
        # TODO: implement the forward pass of neck here
