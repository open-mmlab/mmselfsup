import argparse
import importlib
import numpy as np
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openselfsup.utils import dist_forward_collect, nondist_forward_collect
from openselfsup.datasets import build_dataloader, build_dataset
from openselfsup.models import build_model
from openselfsup.models.utils import MultiPooling
from openselfsup.utils import get_root_logger


class ExtractProcess(object):

    def __init__(self,
                 pool_type='specified',
                 backbone='resnet50',
                 layer_indices=(0, 1, 2, 3, 4)):
        self.multi_pooling = MultiPooling(
            pool_type, in_indices=layer_indices, backbone=backbone)

    def _forward_func(self, model, **x):
        backbone_feats = model(mode='extract', **x)
        pooling_feats = self.multi_pooling(backbone_feats)
        flat_feats = [xx.view(xx.size(0), -1) for xx in pooling_feats]
        feat_dict = {'feat{}'.format(i + 1): feat.cpu() \
            for i, feat in enumerate(flat_feats)}
        return feat_dict

    def extract(self, model, data_loader, distributed=False):
        model.eval()
        func = lambda **x: self._forward_func(model, **x)
        if distributed:
            rank, world_size = get_dist_info()
            results = dist_forward_collect(func, data_loader, rank,
                                           len(data_loader.dataset))
        else:
            results = nondist_forward_collect(func, data_loader,
                                              len(data_loader.dataset))
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='OpenSelfSup extract features of a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--pretrained', default='random',
        help='pretrained model file, exclusive to --checkpoint')
    parser.add_argument(
        '--dataset-config',
        default='benchmarks/extract_info/voc07.py',
        help='extract dataset config file path')
    parser.add_argument(
        '--layer-ind',
        type=str,
        help='layer indices, separated by comma, e.g., "0,1,2,3,4"')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    layer_ind = [int(idx) for idx in args.layer_ind.split(',')]
    cfg.model.backbone.out_indices = layer_ind

    # checkpoint and pretrained are exclusive
    assert args.pretrained == "random" or args.checkpoint is None, \
        "Checkpoint and pretrained are exclusive."

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        for field in ['train', 'val', 'test']:
            if hasattr(cfg.data, field):
                getattr(cfg.data, field).data_source.memcached = False

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'extract_{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset_cfg = mmcv.Config.fromfile(args.dataset_config)
    dataset = build_dataset(dataset_cfg.data.extract)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=dataset_cfg.data.imgs_per_gpu,
        workers_per_gpu=dataset_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # specify pretrained model
    if args.pretrained != 'random':
        assert isinstance(args.pretrained, str)
        cfg.model.pretrained = args.pretrained

    # build the model and load checkpoint
    model = build_model(cfg.model)
    if args.checkpoint is not None:
        logger.info("Use checkpoint: {} to extract features".format(
            args.checkpoint))
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    elif args.pretrained != "random":
        logger.info('Use pretrained model: {} to extract features'.format(
            args.pretrained))
    else:
        logger.info('No checkpoint or pretrained is give, use random init.')
        
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    # build extraction processor
    extractor = ExtractProcess(
        pool_type='specified', backbone='resnet50', layer_indices=layer_ind)

    # run
    outputs = extractor.extract(model, data_loader, distributed=distributed)
    rank, _ = get_dist_info()
    mmcv.mkdir_or_exist("{}/features/".format(args.work_dir))
    if rank == 0:
        for key, val in outputs.items():
            split_num = len(dataset_cfg.split_name)
            split_at = dataset_cfg.split_at
            for ss in range(split_num):
                output_file = "{}/features/{}_{}.npy".format(
                    args.work_dir, dataset_cfg.split_name[ss], key)
                if ss == 0:
                    np.save(output_file, val[:split_at[0]])
                elif ss == split_num - 1:
                    np.save(output_file, val[split_at[-1]:])
                else:
                    np.save(output_file, val[split_at[ss - 1]:split_at[ss]])


if __name__ == '__main__':
    main()
