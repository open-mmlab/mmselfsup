# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.runner import Runner
from torch.utils.data import DataLoader

from mmselfsup.utils import dist_forward_collect, nondist_forward_collect
from .multi_pooling import MultiPooling


class AvgPool2d(nn.Module):
    """The wrapper for AdaptiveAvgPool2d, which supports tuple input."""

    def __init__(self, output_size: int = 1) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function."""
        assert len(x) == 1
        return [self.avgpool(x[-1])]


class Extractor():
    """Feature extractor.

    The extractor support to build its own DataLoader, customized models,
    pooling type. It also has distributed and non-distributed mode.

    Args:
        extract_dataloader (dict): A dict to build DataLoader object.
        seed (int, optional): Random seed. Defaults to None.
        dist_mode (bool): Use distributed extraction or not. Defaults to False.
        pool_cfg (dict, optional): The configs of pooling. Defaults to
            dict(type='AvgPool2d', output_size=1).
    """

    POOL_MAP = {
        'AvgPool2d': AvgPool2d,
        'MultiPooling': MultiPooling,
    }

    def __init__(self,
                 extract_dataloader: Union[DataLoader, dict],
                 seed: Optional[int] = None,
                 dist_mode: bool = False,
                 pool_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        self.data_loader = Runner.build_dataloader(
            dataloader=extract_dataloader, seed=seed)
        self.dist_mode = dist_mode

        # build pooling layer
        self.pool_cfg = pool_cfg
        if pool_cfg is not None:
            self.pool = self.POOL_MAP[pool_cfg.pop('type')](**pool_cfg)
            self.feature_indices = pool_cfg.get('in_indices', [4])

    def _forward_func(self, model: BaseModel,
                      packed_data: List[dict]) -> Dict[str, torch.Tensor]:
        """The forward function to extract features.

        Args:
            model (BaseModel): The model used for extracting features.
            packed_data (List[Dict]): The input data for model.

        Returns:
            Dict[str, torch.Tensor]: The output features.
        """
        # preprocess data
        batch_inputs, batch_data_samples = model.data_preprocessor(packed_data)

        # backbone features
        features = model(batch_inputs, batch_data_samples, mode='tensor')

        # pooling features
        if self.pool_cfg is None:
            features = model.neck([features[-1]])
        else:
            features = self.pool(features)

        # flat features
        flat_features = [feat.view(feat.size(0), -1) for feat in features]

        feature_dict = dict()
        if self.pool_cfg is None:
            feature_dict['feat'] = flat_features[0]
        else:
            for i, feat in enumerate(flat_features):
                feature_dict[f'feat{self.feature_indices[i] + 1}'] = feat
        return feature_dict

    def __call__(self, model: BaseModel) -> Dict[str, torch.Tensor]:
        model.eval()

        # the function sent to collect function
        def func(packed_data):
            return self._forward_func(model, packed_data)

        if self.dist_mode:
            feats = dist_forward_collect(func, self.data_loader,
                                         len(self.data_loader.dataset))
        else:
            feats = nondist_forward_collect(func, self.data_loader,
                                            len(self.data_loader.dataset))
        return feats
