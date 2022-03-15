# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import get_dist_info

from .collect import dist_forward_collect, nondist_forward_collect


def single_gpu_test(model, data_loader):
    model.eval()

    # the function sent to collect function
    def test_mode_func(**x):
        return model(mode='test', **x)

    results = nondist_forward_collect(test_mode_func, data_loader,
                                      len(data_loader.dataset))
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    model.eval()

    # the function sent to collect function
    def test_mode_func(**x):
        return model(mode='test', **x)

    rank, world_size = get_dist_info()
    results = dist_forward_collect(test_mode_func, data_loader, rank,
                                   len(data_loader.dataset))
    return results
