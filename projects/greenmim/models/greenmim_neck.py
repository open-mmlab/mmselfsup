# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcls.models.backbones.vision_transformer import TransformerEncoderLayer
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class GreenMIMNeck(BaseModule):
    """Pre-train Neck For GreenMIM.

    This neck reconstructs the original image from the shrunk feature map.
    """

    def __init__(
            self,
            in_channels: int,
            encoder_stride: int,
            img_size: int,
            patch_size: int,
            embed_dim: int = 96,
            depths: list = [2, 2, 6, 2],
            decoder_embed_dim: int = 512,
            mlp_ratio: float = 4.,
            decoder_depth: int = 8,
            decoder_num_heads: int = 16,
            norm_cfg: dict = dict(type='LN', eps=1e-6),
    ) -> None:
        super().__init__()

        patch_resolution = img_size // patch_size
        num_patches = (patch_resolution // (2**(len(depths) - 1)))**2
        # SwinMAE decoder specifics
        embed_dim = embed_dim * 2**(len(depths) - 1)
        patch_size = patch_size * 2**(len(depths) - 1)
        self.patch_size = patch_size
        self.decoder_embed = nn.Identity(
        ) if embed_dim == decoder_embed_dim else nn.Linear(
            embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

        self.decoder_norm = torch.nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_channels,
            bias=True)  # encoder to decoder

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**.5),
            cls_token=False)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0],
                                             ids_restore.shape[1] - x.shape[1],
                                             1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1,
                                                   x.shape[2]))  # unshuffle

        # add pos embed
        x = x_ + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x


def get_2d_sincos_pos_embed(embed_dim: int,
                            grid_size: int,
                            cls_token: bool = False) -> np.ndarray:
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size,
    embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed],
                                   axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int,
                                      grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2,
                                              grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: list) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
