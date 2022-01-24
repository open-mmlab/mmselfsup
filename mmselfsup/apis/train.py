# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistEvalHook, DistSamplerSeedHook, EvalHook,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmselfsup.core import (DistOptimizerHook, GradAccumFp16OptimizerHook,
                            build_optimizer)
from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.utils import get_root_logger, multi_gpu_test, single_gpu_test


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
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

    data_loaders = [
        build_dataloader(
            ds,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False),
            prefetch=cfg.prefetch,
            persistent_workers=cfg.persistent_workers,
            img_norm_cfg=cfg.img_norm_cfg) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model if next(model.parameters()).is_cuda else model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # build runner
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = GradAccumFp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config or \
            'frozen_layers_cfg' in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register custom hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            if hook_cfg.type == 'DeepClusterHook':
                common_params = dict(dist_mode=True, data_loaders=data_loaders)
            else:
                common_params = dict(dist_mode=True)
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS, common_params)
            runner.register_hook(hook, priority=priority)

    # register evaluation hook
    if cfg.get('evaluation', None):
        val_dataset = build_dataset(cfg.data.val)
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            prefetch=cfg.data.val.prefetch,
            drop_last=getattr(cfg.data, 'drop_last', False),
            img_norm_cfg=cfg.get('img_norm_cfg', dict()))
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        eval_fn = multi_gpu_test if distributed else single_gpu_test
        # `EvalHook` needs to be executed after `IterTimerHook`.
        # Otherwise, it will cause a bug if use `IterBasedRunner`.
        # Refers to https://github.com/open-mmlab/mmcv/issues/1261
        runner.register_hook(
            eval_hook(val_dataloader, test_fn=eval_fn, **eval_cfg),
            priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
