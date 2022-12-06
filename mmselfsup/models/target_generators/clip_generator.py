# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import _load_checkpoint

from mmselfsup.registry import MODELS
from ..utils import build_clip_model


@MODELS.register_module()
class CLIPGenerator(BaseModule):
    """Get the features and attention from the last layer of CLIP.

    This module is used to generate target features in masked image modeling.

    Args:
        tokenizer_path (str): The path of the checkpoint of CLIP.
    """

    def __init__(self, tokenizer_path: str) -> None:
        super().__init__()
        self.tokenizer = build_clip_model(
            _load_checkpoint(tokenizer_path), False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the features and attention from the last layer of CLIP.

        Args:
            x (torch.Tensor): The input image, which is of shape (N, 3, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The features and attention from the last layer of CLIP,
                which are of shape (N, L, C) and (N, L, L), respectively.
        """
        # use the visual branch of CLIP to get the features
        clip_features = self.tokenizer.encode_image(x)
        return clip_features
