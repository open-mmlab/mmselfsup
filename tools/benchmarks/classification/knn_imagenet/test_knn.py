# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.models.utils import ExtractProcess, knn_classifier
from mmselfsup.utils import get_root_logger


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
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
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
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir and init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    knn_work_dir = osp.join(cfg.work_dir, 'knn/')
    mmcv.mkdir_or_exist(osp.abspath(knn_work_dir))
    log_file = osp.join(knn_work_dir, f'knn_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
    dataset_train = build_dataset(dataset_cfg.data.train)
    dataset_val = build_dataset(dataset_cfg.data.val)
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    data_loader_train = build_dataloader(
        dataset_train,
        samples_per_gpu=dataset_cfg.data.samples_per_gpu,
        workers_per_gpu=dataset_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    data_loader_val = build_dataloader(
        dataset_val,
        samples_per_gpu=dataset_cfg.data.samples_per_gpu,
        workers_per_gpu=dataset_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model
    model = build_algorithm(cfg.model)
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

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    model.eval()
    # build extraction processor and run
    extractor = ExtractProcess()
    train_feats = extractor.extract(
        model, data_loader_train, distributed=distributed)['feat']
    val_feats = extractor.extract(
        model, data_loader_val, distributed=distributed)['feat']

    train_feats = torch.from_numpy(train_feats)
    val_feats = torch.from_numpy(val_feats)
    train_labels = torch.LongTensor(dataset_train.data_source.get_gt_labels())
    val_labels = torch.LongTensor(dataset_val.data_source.get_gt_labels())

    logger.info('Features are extracted! Start k-NN classification...')

    rank, _ = get_dist_info()
    if rank == 0:
        if args.use_cuda:
            train_feats = train_feats.cuda()
            val_feats = val_feats.cuda()
            train_labels = train_labels.cuda()
            val_labels = val_labels.cuda()
        for k in args.num_knn:
            top1, top5 = knn_classifier(train_feats, train_labels, val_feats,
                                        val_labels, k, args.temperature)
            logger.info(
                f'{k}-NN classifier result: Top1: {top1}, Top5: {top5}')


if __name__ == '__main__':
    main()
