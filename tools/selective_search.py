# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import logging
import os
import os.path as osp
import time

import mmengine
import torch
import torch.multiprocessing
from mmengine.config import Config, DictAction
from mmengine.dist import get_dist_info, init_dist
from mmengine.logging import MMLogger, print_log
from mmengine.registry import build_model_from_cfg
from mmengine.runner import Runner

from mmselfsup.registry import MODELS
from mmselfsup.utils import register_all_modules

# from mmselfsup.datasets import build_dataloader, build_dataset
# from mmselfsup.models import build_model
# from mmselfsup.utils import (get_root_logger, traverse_replace, print_log)
torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate(bbox, **kwargs):

    if not isinstance(bbox, list):
        bbox = bbox.tolist()
        # dict
    data_ss = {}
    data_ss['bbox'] = bbox
    return data_ss


def nondist_single_forward_collect(func, data_loader, length):
    """Forward and collect network outputs.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        length (int): Expected length of output arrays.

    Returns:
        results_all (dict(list)): The concatenated outputs.
    """
    results = []
    prog_bar = mmengine.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_all[k] = [
            batch[k].squeeze().numpy().tolist() for batch in results
        ]
        assert len(results_all[k]) == length
    return results_all


def dist_single_forward_collect(func, data_loader, rank, length):
    """Forward and collect network outputs in a distributed manner.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        rank (int): This process id.

    Returns:
        results_all (dict(list)): The concatenated outputs.
    """
    results = []
    if rank == 0:
        prog_bar = mmengine.ProgressBar(len(data_loader))
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)  # dict{key: tensor}
        results.append(result)

        if rank == 0:
            prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        results_list = [
            batch[k].squeeze().numpy().tolist() for batch in results
        ]
        results_all[k] = results_list
        # assert len(results_all[k]) == length
    return results_all


def single_gpu_test(model, data_loader):
    model.eval()

    def func(**x):
        return model(mode='test', **x)

    # func = lambda **x: model(mode='test', **x)
    results = nondist_single_forward_collect(func, data_loader,
                                             len(data_loader.dataset))
    return results


def multi_gpu_test(model, data_loader):
    model.eval()

    def func(**x):
        return model(mode='test', **x)

    # func = lambda **x: model(mode='test', **x)
    rank, world_size = get_dist_info()
    results = dist_single_forward_collect(func, data_loader, rank,
                                          len(data_loader.dataset))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        'config',
        default='configs/selfsup/orl/stage2/selective_search.py',
        type=str,
        help='train config file path')
    parser.add_argument(
        'output',
        default='../data/coco/meta/train2017_selective_search_proposal.json ',
        type=str,
        help='output total selective search proposal json file')
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # register all modules in mmselfsup into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # # set cudnn_benchmark
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # # check memcached package exists
    # if importlib.util.find_spec('mc') is None:
    #     traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # if args.launcher == 'slurm':
        # cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir,
                        'SelectiveSearch_{}.log'.format(timestamp))
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
    logging.basicConfig(filename=log_file, level=cfg.log_level)

    logger = MMLogger.get_instance(
        'mmengine', log_file=log_file, log_level=cfg.log_level)

    # build the model
    model = build_model_from_cfg(cfg.model, registry=MODELS)

    # build the dataloader
    # dataset = build_dataset(cfg.data.val)
    # data_loader = build_from_cfg(cfg.val_dataloader,registry=DATASETS)
    data_loader = Runner.build_dataloader(cfg.val_dataloader)

    # outputs = single_gpu_test(model, data_loader)

    if not distributed:
        outputs = single_gpu_test(model, data_loader)
    else:
        outputs = multi_gpu_test(model, data_loader)  # dict{key: list}

    print(type(outputs))
    if isinstance(outputs, dict):
        print(outputs.keys())

    rank, _ = get_dist_info()
    if rank == 0:
        out = evaluate(**outputs)
        with open(args.output, 'w') as f:
            json.dump(out, f)
        print_log(
            'Selective search proposal json file has been saved to: {}'.format(
                args.output),
            logger=logger)


if __name__ == '__main__':
    main()
