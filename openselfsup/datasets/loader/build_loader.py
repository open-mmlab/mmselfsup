import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

#from .sampler import DistributedGroupSampler, DistributedSampler, GroupSampler
from .sampler import DistributedSampler, DistributedGivenIterationSampler
from torch.utils.data import RandomSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     replace=False,
                     seed=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        replace (bool): Replace or not in random shuffle.
            It works on when shuffle is True.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    if dist:
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, replace=replace)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if replace:
            raise NotImplemented
        sampler = RandomSampler(
            dataset) if shuffle else None  # TODO: set replace
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False,
        worker_init_fn=worker_init_fn if seed is not None else None,
        **kwargs)

    return data_loader


def worker_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
