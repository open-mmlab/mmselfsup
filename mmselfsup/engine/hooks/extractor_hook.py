# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from mmengine.dist import is_distributed
from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmselfsup.models.utils import Extractor
from mmselfsup.registry import HOOKS
from mmselfsup.utils import clustering as _clustering


@HOOKS.register_module()
class ExtractorHook(Hook):
    """feature extractor hook.

    This hook includes the global clustering process in DC.

    Args:
        extractor (dict): Config dict for feature extraction.
        clustering (dict): Config dict that specifies the clustering algorithm.
        unif_sampling (bool): Whether to apply uniform sampling.
        reweight (bool): Whether to apply loss re-weighting.
        reweight_pow (float): The power of re-weighting.
        init_memory (bool): Whether to initialize memory banks used in ODC.
            Defaults to False.
        initial (bool): Whether to call the hook initially. Defaults to True.
        interval (int): Frequency of epochs to call the hook. Defaults to 1.
        seed (int, optional): Random seed. Defaults to None.
    """

    def __init__(
            self,
            extract_dataloader: dict,
            normalize=True,
            seed: Optional[int] = None) -> None:
        
        self.dist_mode = is_distributed()
        self.extractor = Extractor(
            extract_dataloader=extract_dataloader,
            seed=seed,
            dist_mode=self.dist_mode,
            pool_cfg=None)
        self.normalize=normalize
        

    def before_run(self, runner):
        self._extract_func(runner)

    def _extract_func(self, runner):
        # step 1: get features
        runner.model.eval()
        features = self.extractor(runner.model.module)['feat']
        if self.normalize:
            features = nn.functional.normalize(torch.from_numpy(features), dim=1)
        # step 2: save features
        if not self.dist_mode or (self.dist_mode and runner.rank == 0):
            np.save(
                "{}/feature_epoch_{}.npy".format(runner.work_dir,
                                                 runner.epoch),
                features.numpy())
            print_log(
                "Feature extraction done!!! total features: {}\tfeature dimension: {}".format(
                    features.size(0), features.size(1)),
                logger='current')
