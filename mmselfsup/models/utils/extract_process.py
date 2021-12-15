# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import get_dist_info

from mmselfsup.utils.collect import (dist_forward_collect,
                                     nondist_forward_collect)
from .multi_pooling import MultiPooling


class ExtractProcess(object):
    """Extraction process for `extract.py` and `tsne_visualization.py` in
    tools.

    Args:
        pool_type (str): Pooling type in :class:`MultiPooling`. Options are
            "adaptive" and "specified". Defaults to "specified".
        backbone (str): Backbone type, now only support "resnet50".
            Defaults to "resnet50".
        layer_indices (Sequence[int]): Output from which stages.
            0 for stem, 1, 2, 3, 4 for res layers. Defaults to (0, 1, 2, 3, 4).
    """

    def __init__(self,
                 pool_type='specified',
                 backbone='resnet50',
                 layer_indices=(0, 1, 2, 3, 4)):
        self.multi_pooling = MultiPooling(
            pool_type, in_indices=layer_indices, backbone=backbone)
        self.layer_indices = layer_indices
        for i in self.layer_indices:
            assert i in [0, 1, 2, 3, 4]

    def _forward_func(self, model, **x):
        """The forward function of extract process."""
        backbone_feats = model(mode='extract', **x)
        pooling_feats = self.multi_pooling(backbone_feats)
        flat_feats = [xx.view(xx.size(0), -1) for xx in pooling_feats]
        feat_dict = {
            f'feat{self.layer_indices[i] + 1}': feat.cpu()
            for i, feat in enumerate(flat_feats)
        }
        return feat_dict

    def extract(self, model, data_loader, distributed=False):
        """The extract function to apply forward function and choose
        distributed or not."""
        model.eval()

        # the function sent to collect function
        def func(**x):
            return self._forward_func(model, **x)

        if distributed:
            rank, world_size = get_dist_info()
            results = dist_forward_collect(func, data_loader, rank,
                                           len(data_loader.dataset))
        else:
            results = nondist_forward_collect(func, data_loader,
                                              len(data_loader.dataset))
        return results
