# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS
from ..utils import build_clip_model


@MODELS.register_module()
class CLIPGenerator(BaseModule):

    def __init__(self, tokenizer_path: str) -> None:
        super().__init__()
        self.tokenizer = build_clip_model(torch.load(tokenizer_path), False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        clip_features = self.tokenizer.encode_image(x)
        return clip_features
