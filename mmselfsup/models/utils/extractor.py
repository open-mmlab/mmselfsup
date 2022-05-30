# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch.nn as nn
from mmengine import Runner
from mmengine.dist import get_rank

from mmselfsup.utils import dist_forward_collect, nondist_forward_collect
from .multi_pooling import MultiPooling


class Extractor():
    """Feature extractor.

    The extractor support to build its own DataLoader, customized models,
    pooling type. It also has distributed and non-distributed mode.

    Args:
        extract_dataloader (Dict): A dict to build DataLoader object.
        seed (int, optional): Random seed. Defaults to None.
        dist_mode (bool, optional): Use distributed extraction or not.
            Defaults to False.
        pool_cfg (Dict, optional): The configs of pooling. Defaults to
            dict(type='AvgPool2d', output_size=1).
    """

    POOL_MAP = {
        'AvgPool2d': nn.AdaptiveAvgPool2d,
        'MultiPooling': MultiPooling,
    }

    def __init__(self,
                 extract_dataloader: Dict,
                 seed: Optional[int] = None,
                 dist_mode: Optional[bool] = False,
                 pool_cfg: Optional[Dict] = dict(
                     type='AvgPool2d', output_size=1),
                 **kwargs):
        self.data_loader = Runner.build_dataloader(
            dataloader=extract_dataloader, seed=seed)
        self.dist_mode = dist_mode

        # build pooling layer
        self.pool_cfg = pool_cfg
        if pool_cfg is not None:
            self.pool = self.POOL_MAP[pool_cfg.pop('type')](**pool_cfg)
            self.layer_indices = pool_cfg.get('layer_indices', [4])

    def _forward_func(self, model: nn.Module, packed_data: List[Dict]) -> Dict:
        """The forward function to extract features.

        Args:
            model (nn.Module): The model used for extracting.
            packed_data (List[Dict]): The input data for model.

        Returns:
            Tuple[torch.Tensor]: The output features.
        """
        # backbone features
        features = model(packed_data, extract=True)

        # pooling features
        if self.pool_cfg is not None:
            features = self.pool([features[-1]])

        # flat features
        if not isinstance(features, (tuple, list)):
            features = [features]
        flat_features = [feat.view(feat.size(0), -1) for feat in features]

        for i, feat in enumerate(flat_features):
            feature_dict = {f'feat{self.layer_indices[i] + 1}': feat.cpu()}
        return feature_dict

    def __call__(self, model):
        model.eval()

        # the function sent to collect function
        def func(packed_data):
            return self._forward_func(model, packed_data)

        if self.dist_mode:
            rank = get_rank()
            feats = dist_forward_collect(
                func, self.data_loader, rank, len(self.dataset), ret_rank=-1)
        else:
            feats = nondist_forward_collect(func, self.data_loader,
                                            len(self.dataset))
        return feats
