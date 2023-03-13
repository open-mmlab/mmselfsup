# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS
import torch.nn as nn


@MODELS.register_module()
class DINOHead(BaseModule):

    def __init__(self) -> None:
        # TODO: implement the initialization function
