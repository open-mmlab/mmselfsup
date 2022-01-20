# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2d sincos position embedding.

    Borrow this code from https://github.com/facebookresearch/mae.
    Args:
        embed_dim (int): The dimension of token.
        grid_size (int): int of the grid height and width.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb
