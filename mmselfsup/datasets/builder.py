# Copyright (c) OpenMMLab. All rights reserved.
import platform
import random
import warnings
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

from .samplers import DistributedSampler
from .utils import PrefetchLoader

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASOURCES = Registry('datasource')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def build_datasource(cfg, default_args=None):
    return build_from_cfg(cfg, DATASOURCES, default_args)


def build_dataset(cfg, default_args=None):
    from .dataset_wrappers import ConcatDataset, RepeatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     imgs_per_gpu=None,
                     samples_per_gpu=None,
                     workers_per_gpu=1,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     replace=False,
                     seed=None,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): (Deprecated, please use samples_per_gpu) Number of
            images on each GPU, i.e., batch size of each GPU. Defaults to None.
        samples_per_gpu (int): Number of images on each GPU, i.e., batch size
            of each GPU. Defaults to None.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU. `persistent_workers` option needs num_workers > 0.
            Defaults to 1.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Defaults to True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Defaults to True.
        replace (bool): Replace or not in random shuffle.
            It works on when shuffle is True. Defaults to False.
        seed (int): set seed for dataloader.
        pin_memory (bool, optional): If True, the data loader will copy Tensors
            into CUDA pinned memory before returning them. Defaults to True.
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Defaults to True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if imgs_per_gpu is None and samples_per_gpu is None:
        raise ValueError(
            'Please inidcate number of images on each GPU, ',
            '"imgs_per_gpu" and "samples_per_gpu" can not be "None" at the ',
            'same time. "imgs_per_gpu" is deprecated, please use ',
            '"samples_per_gpu".')

    if imgs_per_gpu is not None:
        warnings.warn(f'Got "imgs_per_gpu"={imgs_per_gpu} and '
                      f'"samples_per_gpu"={samples_per_gpu}, "imgs_per_gpu"'
                      f'={imgs_per_gpu} is used in this experiments. '
                      'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                      f'{imgs_per_gpu} in this experiments')
        samples_per_gpu = imgs_per_gpu

    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, replace=replace)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        if replace:
            return NotImplemented
        sampler = None  # TODO: set replace
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        kwargs['persistent_workers'] = persistent_workers

    if kwargs.get('prefetch') is not None:
        prefetch = kwargs.pop('prefetch')
        img_norm_cfg = kwargs.pop('img_norm_cfg')
    else:
        prefetch = False
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        **kwargs)

    if prefetch:
        data_loader = PrefetchLoader(data_loader, img_norm_cfg['mean'],
                                     img_norm_cfg['std'])

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Function to initialize each worker.

    The seed of each worker equals to
    ``num_worker * rank + worker_id + user_seed``.

    Args:
        worker_id (int): Id for each worker.
        num_workers (int): Number of workers.
        rank (int): Rank in distributed training.
        seed (int): Random seed.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
