# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time

import torch
from mmengine import Runner
from mmengine.config import Config, DictAction
from mmengine.dist import get_rank, init_dist
from mmengine.logging import MMLogger
from mmengine.model.wrappers import MMDistributedDataParallel, is_model_wrapper
from mmengine.runner import load_checkpoint
from mmengine.utils import mkdir_or_exist

from mmselfsup.evaluation.functional import knn_eval
from mmselfsup.models.utils import Extractor
from mmselfsup.registry import MODELS
from mmselfsup.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='KNN evaluation')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--dataset-config',
        default='configs/benchmarks/classification/knn_imagenet.py',
        help='knn dataset config file path')
    parser.add_argument(
        '--work-dir', type=str, default=None, help='the dir to save results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
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
    # KNN settings
    parser.add_argument(
        '--num-knn',
        default=[10, 20, 100, 200],
        nargs='+',
        type=int,
        help='Number of NN to use. 20 usually works the best.')
    parser.add_argument(
        '--temperature',
        default=0.07,
        type=float,
        help='Temperature used in the voting coefficient.')
    parser.add_argument(
        '--use-cuda',
        default=True,
        type=bool,
        help='Store the features on GPU. Set to False if you encounter OOM')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
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
    knn_work_dir = osp.join(cfg.work_dir, 'knn/')
    mkdir_or_exist(osp.abspath(knn_work_dir))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(knn_work_dir, f'knn_{timestamp}.log')
    logger = MMLogger.get_instance(
        'mmselfsup',
        logger_name='mmselfsup',
        log_file=log_file,
        log_level=cfg.log_level)

    # build dataloader
    dataset_cfg = Config.fromfile(args.dataset_config)
    data_loader_train = Runner.build_dataloader(
        dataloader=dataset_cfg.train_dataloader, seed=args.seed)
    data_loader_val = Runner.build_dataloader(
        dataloader=dataset_cfg.val_dataloader, seed=args.seed)

    # build the model
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
    extractor_train = Extractor(
        extract_dataloader=data_loader_train,
        seed=args.seed,
        dist_mode=distributed,
        pool_cfg=copy.deepcopy(dataset_cfg.pool_cfg))
    extractor_val = Extractor(
        extract_dataloader=data_loader_val,
        seed=args.seed,
        dist_mode=distributed,
        pool_cfg=copy.deepcopy(dataset_cfg.pool_cfg))
    train_feats = extractor_train(model)['feat5']
    val_feats = extractor_val(model)['feat5']

    train_feats = torch.from_numpy(train_feats)
    val_feats = torch.from_numpy(val_feats)
    train_labels = torch.LongTensor(data_loader_train.dataset.get_gt_labels())
    val_labels = torch.LongTensor(data_loader_val.dataset.get_gt_labels())

    logger.info('Features are extracted! Start k-NN classification...')

    # run knn
    rank = get_rank()
    if rank == 0:
        if args.use_cuda:
            train_feats = train_feats.cuda()
            val_feats = val_feats.cuda()
            train_labels = train_labels.cuda()
            val_labels = val_labels.cuda()
        for k in args.num_knn:
            top1, top5 = knn_eval(train_feats, train_labels, val_feats,
                                  val_labels, k, args.temperature)
            logger.info(
                f'{k}-NN classifier result: Top1: {top1}, Top5: {top5}')


if __name__ == '__main__':
    main()
