# Copyright (c) OpenMMLab. All rights reserved.
import mmengine
import torch
from mmengine.dist import collect_results_gpu, get_rank
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
    rank = get_rank()
    results = []
    if rank == 0:
        prog_bar = mmengine.ProgressBar(len(data_loader))
    for _, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(data)  # dict{key: tensor}

        # batch_dict = {}
        # for k in result.keys():
        #     batch_local = []
        #     batch_local.extend(result[k].tolist())
        #     batch_gathered = collect_results_gpu(
        #         batch_local,
        #         len(batch_local) * get_world_size())

        #     if rank == 0:
        #         batch_dict[k] = torch.Tensor(
        #             batch_gathered, device=torch.device('cuda:0'))

        results.append(result)

        if rank == 0:
            prog_bar.update()

    results_dict = {}
    for k in results[0].keys():
        results_local = []
        for batch in results:
            results_local.extend(batch[k].tolist())
        results_gathered = collect_results_gpu(results_local, length)
        if rank == 0:
            results_dict[k] = torch.Tensor(results_gathered).to(
                torch.device('cuda:0'))
    return results_dict
