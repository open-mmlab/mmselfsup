# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from mmengine.dist import all_gather


def gather_tensors(input_array: np.ndarray) -> np.ndarray:
    """Gather tensor from all GPUs.

    Args:
        inpyt_array (np.ndarray): Extracted features.

    Returns:
        np.ndarray: Gatherd features.
    """
    world_size = dist.get_world_size()

    # gather shapes first
    myshape = input_array.shape
    mycount = input_array.size
    shape_tensor = torch.Tensor(np.array(myshape)).cuda()
    all_shape = [
        torch.Tensor(np.array(myshape)).cuda() for i in range(world_size)
    ]
    dist.all_gather(all_shape, shape_tensor)

    # compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)

    # padding tensors and gather them
    output_tensors = [
        torch.Tensor(max_count).cuda() for i in range(world_size)
    ]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = input_array.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).cuda()
    dist.all_gather(output_tensors, input_tensor)

    # unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [
        x[:all_count[i]].reshape(all_shape[i])
        for i, x in enumerate(padded_output)
    ]
    return output


def gather_tensors_batch(input_array: np.ndarray,
                         part_size: Optional[int] = 100,
                         ret_rank: Optional[int] = -1) -> List[np.ndarray]:
    """Batch-wise gathering to avoid CUDA out of memory.

    Args:
        input_array (np.ndarray): Extracted features.
        part_size (int, optional): The defined part size to separate batch.
            Defaults to 100.
        ret_rank (int, optional): The process that returns. Other processes
            will return None. Defaults to -1.

    Returns:
        np.ndarray: Gatherd features.
    """
    rank = dist.get_rank()
    all_features = []
    part_num = input_array.shape[0] // part_size + 1 if input_array.shape[
        0] % part_size != 0 else input_array.shape[0] // part_size
    for i in range(part_num):
        part_feat = input_array[i *
                                part_size:min((i + 1) *
                                              part_size, input_array.shape[0]),
                                ...]
        assert part_feat.shape[
            0] > 0, f'rank: {rank}, length of part features should > 0'
        gather_part_feat = gather_tensors(part_feat)
        all_features.append(gather_part_feat)
    if ret_rank == -1:
        all_features = [
            np.concatenate([all_features[i][j] for i in range(part_num)],
                           axis=0) for j in range(len(all_features[0]))
        ]
        return all_features
    else:
        if rank == ret_rank:
            all_features = [
                np.concatenate([all_features[i][j] for i in range(part_num)],
                               axis=0) for j in range(len(all_features[0]))
            ]
            return all_features
        else:
            return None


@torch.no_grad()
def concat_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Performs all_gather operation on the provided tensors.

    Args:
        tensor (torch.Tensor): Tensor to be broadcast from current process.

    Returns:
        torch.Tensor: The concatnated tensor.
    """
    tensors_gather = all_gather(tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output
