# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.dist import all_gather


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
