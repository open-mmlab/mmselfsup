# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from .gather import concat_all_gather


@torch.no_grad()
def batch_shuffle_ddp(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch shuffle, for making use of BatchNorm.

    *** Only support DistributedDataParallel (DDP) model. ***

    Args:
        x (torch.Tensor): Data in each GPU.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Output of shuffle operation.
            - x_gather[idx_this]: Shuffled data.
            - idx_unshuffle: Index for restoring.
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle


@torch.no_grad()
def batch_unshuffle_ddp(x: torch.Tensor,
                        idx_unshuffle: torch.Tensor) -> torch.Tensor:
    """Undo batch shuffle.

    *** Only support DistributedDataParallel (DDP) model. ***

    Args:
        x (torch.Tensor): Data in each GPU.
        idx_unshuffle (torch.Tensor): Index for restoring.

    Returns:
        torch.Tensor: Output of unshuffle operation.
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]
