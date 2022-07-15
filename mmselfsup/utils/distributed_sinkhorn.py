# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

# This file is modified from
# https://github.com/facebookresearch/swav/blob/main/main_swav.py
import torch
from mmengine.dist import all_reduce


@torch.no_grad()
def distributed_sinkhorn(out: torch.Tensor, sinkhorn_iterations: int,
                         world_size: int, epsilon: float) -> torch.Tensor:
    """Apply the distributed sinknorn optimization on the scores matrix to find
    the assignments.

    Args:
        out (torch.Tensor): The scores matrix
        sinkhorn_iterations (int): Number of iterations in Sinkhorn-Knopp
            algorithm.
        world_size (int): The world size of the process group.
        epsilon (float): regularization parameter for Sinkhorn-Knopp algorithm.

    Returns:
        torch.Tensor: Output of sinkhorn algorithm.
    """
    eps_num_stab = 1e-12
    Q = torch.exp(out / epsilon).t(
    )  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        u = torch.sum(Q, dim=1, keepdim=True)
        if len(torch.nonzero(u == 0)) > 0:
            Q += eps_num_stab
            u = torch.sum(Q, dim=1, keepdim=True, dtype=Q.dtype)
            all_reduce(u)
        Q /= u
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()
