# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import pseudo_collate, worker_init_fn
from mmengine.dist import get_rank, init_dist
from mmengine.logging import MMLogger
from mmengine.model.wrappers import MMDistributedDataParallel, is_model_wrapper
from mmengine.runner import load_checkpoint
from mmengine.utils import mkdir_or_exist
from torch.utils.data import DataLoader

from mmselfsup.models.utils import Extractor
from mmselfsup.registry import DATA_SAMPLERS, DATASETS, MODELS
from mmselfsup.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSelfSup extract features of a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--dataset-config',
        default='configs/benchmarks/classification/svm_voc07.py',
        help='extract dataset config file path')
    parser.add_argument(
        '--layer-ind',
        type=str,
        help='layer indices, separated by comma, e.g., "0,1,2,3,4"')
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
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
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # register all modules in mmselfsup into the registries
    register_all_modules()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.env_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher)

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'extract_{timestamp}.log')
    logger = MMLogger.get_instance(
        'mmselfsup',
        logger_name='mmselfsup',
        log_file=log_file,
        log_level=cfg.log_level)

    # build the dataset
    dataset_cfg = Config.fromfile(args.dataset_config)
    extract_dataloader_cfg = dataset_cfg.get('extract_dataloader')
    extract_dataset_cfg = extract_dataloader_cfg.pop('extract_dataset')
    if isinstance(extract_dataset_cfg, dict):
        dataset = DATASETS.build(extract_dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()

    # build sampler
    sampler_cfg = extract_dataloader_cfg.pop('sampler')
    if isinstance(sampler_cfg, dict):
        sampler = DATA_SAMPLERS.build(
            sampler_cfg, default_args=dict(dataset=dataset, seed=args.seed))

    # build dataloader
    init_fn: Optional[partial]
    if args.seed is not None:
        init_fn = partial(
            worker_init_fn,
            num_workers=extract_dataloader_cfg.get('num_workers'),
            rank=get_rank(),
            seed=args.seed)
    else:
        init_fn = None

    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=pseudo_collate,
        worker_init_fn=init_fn,
        **extract_dataloader_cfg)

    # build the model
    # get out_indices from args
    layer_ind = [int(idx) for idx in args.layer_ind.split(',')]
    cfg.model.backbone.out_indices = layer_ind
    model = MODELS.build(cfg.model)
    model.init_weights()

    # model is determined in this priority: init_cfg > checkpoint > random
    if hasattr(cfg.model.backbone, 'init_cfg'):
        if getattr(cfg.model.backbone.init_cfg, 'type', None) == 'Pretrained':
            logger.info(
                f'Use pretrained model: '
                f'{cfg.model.backbone.init_cfg.checkpoint} to extract features'
            )
    elif args.checkpoint is not None:
        logger.info(f'Use checkpoint: {args.checkpoint} to extract features')
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        logger.info('No pretrained or checkpoint is given, use random init.')

    if torch.cuda.is_available():
        model = model.cuda()

    if distributed:
        model = MMDistributedDataParallel(
            module=model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    if is_model_wrapper(model):
        model = model.module

    # build extractor and extract features
    extractor = Extractor(
        extract_dataloader=data_loader,
        seed=args.seed,
        dist_mode=distributed,
        pool_cfg=dataset_cfg.pool_cfg)
    outputs = extractor(model)

    # run
    rank = get_rank()
    mkdir_or_exist(f'{cfg.work_dir}/features/')
    if rank == 0:
        for key, val in outputs.items():
            split_num = len(dataset_cfg.split_name)
            split_at = dataset_cfg.split_at
            for ss in range(split_num):
                output_file = f'{cfg.work_dir}/features/' \
                              f'{dataset_cfg.split_name[ss]}_{key}.npy'
                if ss == 0:
                    np.save(output_file, val[:split_at[0]])
                elif ss == split_num - 1:
                    np.save(output_file, val[split_at[-1]:])
                else:
                    np.save(output_file, val[split_at[ss - 1]:split_at[ss]])


if __name__ == '__main__':
    main()
