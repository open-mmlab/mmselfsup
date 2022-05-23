# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from mmengine import Runner

from mmselfsup.utils import dist_forward_collect, nondist_forward_collect


class Extractor():
    """Feature extractor.

    Args:
        extract_dataloader (dict): A dict to build Dataloader object.
        seed (int, optional): Random seed. Defaults to None.
        dist_mode (bool, optional): Use distributed extraction or not.
            Defaults to False.
    """

    def __init__(self,
                 extract_dataloader: Dict,
                 seed: Optional[int] = None,
                 dist_mode: bool = False,
                 **kwargs):
        self.data_loader = Runner.build_dataloader(
            extract_dataloader=extract_dataloader, seed=seed)
        self.dist_mode = dist_mode

    def _forward_func(self, runner, packed_data):
        backbone_feat = runner.model(packed_data, extract=True)
        last_layer_feat = runner.model.module.neck([backbone_feat[-1]])[0]
        last_layer_feat = last_layer_feat.view(last_layer_feat.size(0), -1)
        return dict(feature=last_layer_feat.cpu())

    def __call__(self, runner):
        # the function sent to collect function
        def func(packed_data):
            return self._forward_func(runner, packed_data)

        if self.dist_mode:
            feats = dist_forward_collect(
                func,
                self.data_loader,
                runner.rank,
                len(self.dataset),
                ret_rank=-1)['feature']  # NxD
        else:
            feats = nondist_forward_collect(func, self.data_loader,
                                            len(self.dataset))['feature']
        return feats
