# This file is modified from
# https://github.com/facebookresearch/swav/blob/main/main_swav.py

import torch
import torch.distributed as dist


@torch.no_grad()
def distributed_sinkhorn(out, sinkhorn_iterations, world_size, epsilon):
    """Apply the distributed sinknorn optimization on the scores matrix to find
    the assignments."""
    eps_num_stab = 1e-12
    Q = torch.exp(out / epsilon).t(
    )  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        u = torch.sum(Q, dim=1, keepdim=True)
        if len(torch.nonzero(u == 0)) > 0:
            Q += eps_num_stab
            u = torch.sum(Q, dim=1, keepdim=True, dtype=Q.dtype)
            if dist.is_initialized():
                dist.all_reduce(u)
        Q /= u
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()
