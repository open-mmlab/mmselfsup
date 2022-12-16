# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import torch
from mmengine.dist import collect_results_gpu, get_dist_info
from torch.utils.data import DataLoader


def nondist_forward_collect(func: object, data_loader: DataLoader,
                            length: int) -> dict:
    """Forward and collect network outputs.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data.
        data_loader (DataLoader): the torch DataLoader to yield data.
        length (int): Expected length of output arrays.

    Returns:
        Dict[str, torch.Tensor]: The concatenated outputs.
    """
    results = []
    prog_bar = mmengine.ProgressBar(len(data_loader))
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(data)  # dict{key: tensor}
        results.append(result)
        prog_bar.update()

    results_dict = {}
    for k in results[0].keys():
        results_dict[k] = torch.cat([batch[k] for batch in results], dim=0)
        assert results_dict[k].size(0) == length
    return results_dict


def dist_forward_collect(func: object, data_loader: DataLoader,
                         length: int) -> dict:
    """Forward and collect network outputs in a distributed manner.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data.
        data_loader (DataLoader): the torch DataLoader to yield data.
        length (int): Expected length of output arrays.

    Returns:
        Dict[str, torch.Tensor]: The collected outputs.
    """
    rank, world_size = get_dist_info()
    results = []
    if rank == 0:
        prog_bar = mmengine.ProgressBar(len(data_loader))
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            batch_result = func(data)  # dict{key: tensor}

        # gather batch results to avoid CUDA OOM
        batch_dict = {}
        for k in batch_result.keys():
            batch_local = batch_result[k].tolist()
            batch_gathered = collect_results_gpu(batch_local,
                                                 len(batch_local) * world_size)
            batch_dict[k] = batch_gathered

        results.append(batch_dict)
        if rank == 0:
            prog_bar.update()

    # concat results and convert to tensor
    results_dict = {}
    if rank == 0:
        for k in results[0].keys():
            result = []
            for res in results:
                result.extend(res[k])
            results_dict[k] = torch.Tensor(result[:length]).to(
                torch.device('cuda:0'))
    return results_dict
