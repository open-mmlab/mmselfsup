# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import logging
import os
import os.path as osp
import time

import mmengine
import torch
from mmengine.config import Config
from mmengine.dist import get_dist_info, init_dist
from mmengine.logging import MMLogger, print_log
from mmengine.model import MMDistributedDataParallel
from mmengine.registry import build_model_from_cfg
from mmengine.runner import Runner, load_checkpoint

from mmselfsup.registry import MODELS
from mmselfsup.utils import register_all_modules


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
        if k == 'intra_bbox':
            intra_results_list = [
                batch[k].numpy().tolist() for batch in results
            ]
            results_all[k] = intra_results_list
            assert len(results_all[k]) == length
    inter_results_list = []
    for batch in results:
        merge_batch_results = []
        for k in batch.keys():
            if k != 'intra_bbox':
                merge_batch_results.append(batch[k].numpy().tolist())
        inter_results_list.append(merge_batch_results)
    results_all['inter_bbox'] = inter_results_list
    assert len(results_all['inter_bbox']) == length
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
        if k == 'intra_bbox':
            intra_results_list = [
                batch[k].numpy().tolist() for batch in results
            ]
            results_all[k] = intra_results_list
    inter_results_list = []
    for batch in results:
        merge_batch_results = []
        for k in batch.keys():
            if k != 'intra_bbox':
                merge_batch_results.append(batch[k].numpy().tolist())
        inter_results_list.append(merge_batch_results)
    results_all['inter_bbox'] = inter_results_list
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
    parser = argparse.ArgumentParser(
        description='Generate correspondence in Stage 2 of ORL')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input', type=str, help='input knn instance json file')
    parser.add_argument(
        'output', type=str, help='output correspondence json file')
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
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--port',
        type=int,
        default=29500,
        help='port only works when launcher=="slurm"')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def evaluate(
    json_file,
    dataset_info,
    intra_bbox,
    inter_bbox,
):
    assert (len(intra_bbox) == len(inter_bbox)), \
        'Mismatch the number of images in part training set, \
            got: intra: {} inter: {}'\
            .format(len(intra_bbox), len(inter_bbox))
    data = mmengine.load(json_file)
    # dict
    data_new = {}
    # sub-dict
    info = {}
    image_info = {}
    pseudo_anno = {}
    info['bbox_min_size'] = dataset_info['min_size']
    info['bbox_max_aspect_ratio'] = dataset_info['max_ratio']
    info['bbox_max_iou'] = dataset_info['max_iou_thr']
    info['intra_bbox_num'] = dataset_info['topN']
    info['knn_image_num'] = dataset_info['knn_image_num']
    info['knn_bbox_pair_ratio'] = dataset_info['topk_bbox_ratio']
    image_info['file_name'] = data['images']['file_name']
    image_info['id'] = data['images']['id']
    pseudo_anno['image_id'] = data['pseudo_annotations']['image_id']
    pseudo_anno['bbox'] = intra_bbox
    pseudo_anno['knn_image_id'] = data['pseudo_annotations']['knn_image_id']
    pseudo_anno['knn_bbox_pair'] = inter_bbox
    data_new['info'] = info
    data_new['images'] = image_info
    data_new['pseudo_annotations'] = pseudo_anno
    return data_new


def main():

    args = parse_args()
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])
    # if args.work_dir is not None:
    #     if not os.path.exists(args.work_dir):
    #         os.makedirs(args.work_dir)
    #     cfg.work_dir = args.work_dir

    # ensure to use checkpoint rather than pretraining
    cfg.model.pretrained = None

    # init distributed env first,
    # since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # if args.launcher == 'slurm':
        #     cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'correpondece_{}.log'.format(timestamp))
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
        # os.mkdir(cfg.work_dir)
    logging.basicConfig(filename=log_file, level=cfg.log_level)

    logger = MMLogger.get_instance(
        'mmengine', log_file=log_file, log_level=cfg.log_level)

    # build the model
    model = build_model_from_cfg(cfg.model, registry=MODELS)

    # build the dataloader
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        # model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model.cuda(), data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader)  # dict{key: list}

    rank, _ = get_dist_info()
    if rank == 0:
        out = evaluate(args.input, cfg.dataset_dict, **outputs)
        with open(args.output, 'w') as f:
            json.dump(out, f)
        print_log(
            'Correspondence json file has been saved to: {}'.format(
                args.output),
            logger=logger)


if __name__ == '__main__':
    main()
