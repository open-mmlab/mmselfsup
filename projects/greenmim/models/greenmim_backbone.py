# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.utils import to_2tuple
from torch.nn import functional as F

from mmselfsup.registry import MODELS


class Mlp(BaseModule):

    def __init__(self,
                 in_features: torch.Tensor,
                 hidden_features: int = None,
                 out_features: int = None,
                 act_layer: nn.Module = nn.GELU,
                 drop: float = 0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_coordinates(h: int, w: int, device: str = 'cpu') -> torch.Tensor:
    coords_h = torch.arange(h, device=device)
    coords_w = torch.arange(w, device=device)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    return coords


class WindowAttention(BaseModule):
    r""" Window based multi-head self attention (W-MSA) module with
    relative position bias. It supports both of shifted and
    non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention
            weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output.
            Default: 0.0
    """

    def __init__(self,
                 dim: int,
                 window_size: tuple,
                 num_heads: int,
                 qkv_bias: bool = True,
                 qk_scale: bool = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.) -> None:

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each
        # token inside the window
        # NOTE: the index is not used at pretraining and
        # is kept for compatibility
        coords = get_coordinates(*window_size)  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,  # 2, Wh*Ww, Wh*Ww
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index',
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None,
                pos_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww,
                Wh*Ww) or None
        """
        # projection
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N

        # relative position bias
        assert pos_idx.dim(
        ) == 3, 'Expect the pos_idx/mask to be a 3-d tensor,'
        f'but got{pos_idx.dim()}'
        rel_pos_mask = torch.masked_fill(
            torch.ones_like(mask), mask=mask.bool(), value=0.0)
        pos_idx_m = torch.masked_fill(pos_idx, mask.bool(), value=0).view(-1)
        relative_position_bias = self.relative_position_bias_table[
            pos_idx_m].view(-1, N, N, self.num_heads)  # nW, Wh*Ww, Wh*Ww,nH
        relative_position_bias = relative_position_bias * rel_pos_mask.view(
            -1, N, N, 1)

        nW = relative_position_bias.shape[0]
        relative_position_bias = relative_position_bias.permute(
            0, 3, 1, 2).contiguous()  # nW, nH, Wh*Ww, Wh*Ww
        attn = attn.view(B_ // nW, nW, self.num_heads, N,
                         N) + relative_position_bias.unsqueeze(0)

        # attention mask
        attn = attn + mask.view(1, nW, 1, N, N)
        attn = attn.view(B_, self.num_heads, N, N)

        # normalization
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # aggregation
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, \
            num_heads={self.num_heads}'


class SwinTransformerBlock(BaseModule):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self,
                 dim: int,
                 input_resolution: Tuple[int],
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: bool = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution,
            # we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, \
            'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor,
                rel_pos_idx: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        # W-MSA/SW-MSA, B*nW, N_vis, C
        x = self.attn(x, mask=attn_mask, pos_idx=rel_pos_idx)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, '
        'num_heads={self.num_heads}, ' f'window_size={self.window_size}, '
        f'shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'


class PatchMerging(BaseModule):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.
            Default: nn.LayerNorm
    """

    def __init__(self,
                 input_resolution: Tuple[int],
                 dim: int,
                 norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor,
                mask_prev: torch.Tensor) -> torch.Tensor:
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        ratio = H // 7 if H % 7 == 0 else H // 6  # FIXME
        x = x.view(B, -1, ratio // 2, 2, ratio // 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, L // 4, 4 * C)

        # merging by a linear layer
        x = self.norm(x)
        x = self.reduction(x)

        mask_new = mask_prev.view(1, -1, ratio // 2, 2, ratio // 2,
                                  2).sum(dim=(3, 5))
        assert torch.unique(mask_new).shape[0] == 2  # should be [0, 4]
        mask_new = (mask_new > 0).reshape(1, -1)
        coords_new = get_coordinates(H // 2, W // 2,
                                     x.device).reshape(1, 2, -1)
        coords_new = coords_new.transpose(2, 1)[mask_new].reshape(1, -1, 2)
        return x, coords_new, mask_new

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'


def knapsack(W: int, wt: Tuple[int]) -> Tuple[List[List[int]], list]:
    '''Args:
        W (int): capacity
        wt (tuple[int]): the numbers of elements within each window
    '''
    val = wt
    n = len(val)
    K = [[0 for w in range(W + 1)] for i in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]],
                              K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # stores the result of Knapsack
    res = res_ret = K[n][W]

    # stores the selected indexes
    w = W
    idx = []
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # Either the result comes from the top (K[i-1][w])
        # or from (val[i-1] + K[i-1] [w-wt[i-1]]) as in Knapsack table.
        # If it comes from the latter one, it means the item is included.
        if res == K[i - 1][w]:
            continue
        else:
            # This item is included.
            idx.append(i - 1)
            # Since this weight is included, its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]

    return res_ret, idx[::-1]  # make the idx in an increasing order


def group_windows(group_size: int,
                  num_ele_win: List[int]) -> Tuple[List[int], List[List[int]]]:
    """Greedily apply the DP algorithm to group the elements.

    Args:
        group_size (int): maximal size of the group
        num_ele_win (list[int]): number of visible elements of each window
    Outputs:
        num_ele_group (list[int]): number of elements of each group
        grouped_idx (list[list[int]]): the selected indeices of each group
    """
    wt = num_ele_win.copy()
    ori_idx = list(range(len(wt)))
    grouped_idx = []
    num_ele_group = []

    while len(wt) > 0:
        res, idx = knapsack(group_size, wt)
        num_ele_group.append(res)

        # append the selected idx
        selected_ori_idx = [ori_idx[i] for i in idx]
        grouped_idx.append(selected_ori_idx)

        # remaining idx
        wt = [wt[i] for i in range(len(ori_idx)) if i not in idx]
        ori_idx = [ori_idx[i] for i in range(len(ori_idx)) if i not in idx]

    return num_ele_group, grouped_idx


class GroupingModule(BaseModule):

    def __init__(self,
                 window_size: int,
                 shift_size: int,
                 group_size: int = None) -> None:
        self.window_size = window_size
        self.shift_size = shift_size
        assert shift_size >= 0 and shift_size < window_size

        self.group_size = group_size or self.window_size**2
        self.attn_mask = None
        self.rel_pos_idx = None

    def _get_group_id(self, coords: torch.Tensor) -> torch.Tensor:
        group_id = coords.clone()
        group_id += (self.window_size - self.shift_size) % self.window_size
        group_id = group_id // self.window_size
        group_id = group_id[0, :, 0] * group_id.shape[1] + group_id[
            0, :, 1]  # (N_vis, )
        return group_id

    def _get_attn_mask(self, group_id: torch.Tensor) -> torch.Tensor:
        pos_mask = (group_id == -1)
        pos_mask = torch.logical_and(pos_mask[:, :, None], pos_mask[:,
                                                                    None, :])
        gid = group_id.float()
        attn_mask_float = gid.unsqueeze(2) - gid.unsqueeze(1)
        attn_mask = torch.logical_or(attn_mask_float != 0, pos_mask)
        attn_mask_float.masked_fill_(attn_mask, -100.)
        return attn_mask_float

    def _get_rel_pos_idx(self, coords: torch.Tensor) -> torch.Tensor:
        # num_groups, group_size, group_size, 2
        rel_pos_idx = coords[:, :, None, :] - coords[:, None, :, :]
        rel_pos_idx += self.window_size - 1
        rel_pos_idx[..., 0] *= 2 * self.window_size - 1
        rel_pos_idx = rel_pos_idx.sum(dim=-1)
        return rel_pos_idx

    def _prepare_masking(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, N_vis, 2)
        group_id = self._get_group_id(coords)  # (N_vis, )
        attn_mask = self._get_attn_mask(group_id.unsqueeze(0))
        rel_pos_idx = self._get_rel_pos_idx(coords[:1])

        # do not shuffle
        self.idx_shuffle = None
        self.idx_unshuffle = None

        return attn_mask, rel_pos_idx

    def _prepare_grouping(
            self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # find out and merge the elements within each local window
        # coords: (B, N_vis, 2)
        group_id = self._get_group_id(coords)  # (N_vis, )
        idx_merge = torch.argsort(group_id)
        group_id = group_id[idx_merge].contiguous()
        exact_win_sz = torch.unique_consecutive(
            group_id, return_counts=True)[1].tolist()

        # group the windows by DP algorithm
        self.group_size = min(self.window_size**2, max(exact_win_sz))
        num_ele_group, grouped_idx = group_windows(self.group_size,
                                                   exact_win_sz)

        # pad the splits if their sizes are smaller than the group size
        idx_merge_spl = idx_merge.split(exact_win_sz)
        group_id_spl = group_id.split(exact_win_sz)
        shuffled_idx, attn_mask = [], []
        for num_ele, gidx in zip(num_ele_group, grouped_idx):
            pad_r = self.group_size - num_ele
            # shuffle indexes: (group_size)
            sidx = torch.cat([idx_merge_spl[i] for i in gidx], dim=0)
            shuffled_idx.append(F.pad(sidx, (0, pad_r), value=-1))
            # attention mask: (group_size)
            amask = torch.cat([group_id_spl[i] for i in gidx], dim=0)
            attn_mask.append(F.pad(amask, (0, pad_r), value=-1))

        # shuffle indexes: (num_groups * group_size, )
        self.idx_shuffle = torch.cat(shuffled_idx, dim=0)
        # unshuffle indexes that exclude the padded indexes: (N_vis, )
        self.idx_unshuffle = torch.argsort(
            self.idx_shuffle)[-sum(num_ele_group):]
        self.idx_shuffle[self.idx_shuffle ==
                         -1] = 0  # index_select does not permit negative index

        # attention mask: (num_groups, group_size, group_size)
        attn_mask = torch.stack(attn_mask, dim=0)
        attn_mask = self._get_attn_mask(attn_mask)

        # relative position indexes: (num_groups, group_size, group_size)
        coords_shuffled = coords[0][self.idx_shuffle].reshape(
            -1, self.group_size, 2)
        rel_pos_idx = self._get_rel_pos_idx(
            coords_shuffled)  # num_groups, group_size, group_size
        rel_pos_mask = torch.ones_like(rel_pos_idx).masked_fill_(
            attn_mask.bool(), 0)
        rel_pos_idx = rel_pos_idx * rel_pos_mask

        return attn_mask, rel_pos_idx

    def prepare(self, coords: torch.Tensor,
                mode: torch.Tensor) -> torch.Tensor:
        self._mode = mode
        if mode == 'masking':
            return self._prepare_masking(coords)
        elif mode == 'grouping':
            return self._prepare_grouping(coords)
        else:
            raise KeyError('')

    def group(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == 'grouping':
            self.ori_shape = x.shape
            x = torch.index_select(x, 1, self.idx_shuffle)  # (B, nG*GS, C)
            x = x.reshape(-1, self.group_size, x.shape[-1])  # (B*nG, GS, C)
        return x

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        if self._mode == 'grouping':
            B, N, C = self.ori_shape
            x = x.reshape(B, -1, C)  # (B, nG*GS, C)
            x = torch.index_select(x, 1, self.idx_unshuffle)  # (B, N, C)
        return x


class BasicLayer(BaseModule):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
          value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
            Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default:
            nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at
            the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False.
    """

    def __init__(self,
                 dim: int,
                 input_resolution: Tuple[int],
                 depth: int,
                 num_heads: int,
                 window_size: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: bool = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 downsample: nn.Module = None,
                 use_checkpoint: bool = False) -> None:

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution,
            # we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        else:
            self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self,
                x: torch.Tensor,
                coords: torch.Tensor,
                patch_mask: torch.Tensor,
                return_x_before_down: bool = False) -> torch.Tensor:
        # prepare the attention mask
        # when the number of visible patches is small,
        # all patches are partitioned into a single group
        mode = 'masking' if x.shape[
            1] <= 2 * self.window_size**2 else 'grouping'

        group_block = GroupingModule(self.window_size, 0)
        mask, pos_idx = group_block.prepare(coords, mode)
        if self.window_size < min(
                self.input_resolution) and self.shift_size != 0:
            group_block_shift = GroupingModule(self.window_size,
                                               self.shift_size)
            mask_shift, pos_idx_shift = group_block_shift.prepare(coords, mode)
        else:
            # do not shift
            group_block_shift = group_block
            mask_shift, pos_idx_shift = mask, pos_idx

        # forward with grouping/masking
        for i, blk in enumerate(self.blocks):
            gblk = group_block if i % 2 == 0 else group_block_shift
            attn_mask = mask if i % 2 == 0 else mask_shift
            rel_pos_idx = pos_idx if i % 2 == 0 else pos_idx_shift
            x = gblk.group(x)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, rel_pos_idx)
            else:
                x = blk(x, attn_mask, rel_pos_idx)
            x = gblk.merge(x)

        # patch merging
        if self.downsample is not None:
            x_down, coords, patch_mask = self.downsample(x, patch_mask)
        else:
            x_down = x

        if return_x_before_down:
            return x, x_down, coords, patch_mask
        else:
            return x_down, coords, patch_mask

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution},' \
                f'window_size={self.window_size},' \
                f'shift_size={self.shift_size}, depth={self.depth}'


class PatchEmbed(BaseModule):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels.
            Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 norm_layer: nn.Module = None) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model" \
               f'({self.img_size[0]}*{self.img_size[1]}).'
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


@MODELS.register_module()
class GreenMIMSwinTransformer(BaseBackbone):
    r"""GreenMIM with SwinTransformer backbone.
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head.
            Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
            Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
            Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the
            patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding.
            Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory.
            Default: False
    """

    def __init__(self,
                 arch: str = 'B',
                 stage_cfgs: dict = None,
                 img_size: int = 224,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 depths: list = [2, 2, 6, 2],
                 num_heads: list = [3, 6, 12, 24],
                 window_size: int = 7,
                 mlp_ratio: float = 4.,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 norm_pix_loss: bool = False,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 ape: bool = False,
                 patch_norm: bool = True,
                 drop_path_rate: float = 0.1,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 use_checkpoint: bool = False,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        # SwinTransformer specifics
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer),
                                  patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if
                (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        num_patches = np.prod(self.layers[-1].input_resolution)
        self.num_patches = num_patches
        patch_size = patch_size * 2**(len(depths) - 1)
        self.final_patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def init_weights(self):
        # initialization
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        super().init_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self,
                   x: torch.Tensor,
                   patch_size: int = None) -> torch.Tensor:
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size or self.final_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(
            self, x: torch.Tensor,
            mask_ratio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L = 1, self.num_patches  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask.scatter_add_(
            1, ids_keep,
            torch.full([N, len_keep],
                       fill_value=-1,
                       dtype=mask.dtype,
                       device=x.device))
        assert (mask.gather(1, ids_shuffle).gather(1,
                                                   ids_restore) == mask).all()

        # repeat the mask
        ids_restore = ids_restore.repeat(x.shape[0], 1)
        mask = mask.repeat(x.shape[0], 1)

        return mask, ids_restore

    def forward_features(self, x: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        # patch embedding
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # mask out some patches according to the random mask
        B, N, C = x.shape
        H, W = self.patches_resolution
        ratio = N // mask.shape[1]
        mask = mask[:1].clone()  # we use the same mask for the whole batch
        assert ratio * mask.shape[1] == N
        window_size = int(ratio**0.5)
        if ratio > 1:  # mask_size != patch_embed_size
            Mh, Mw = [sz // window_size for sz in self.patches_resolution]
            mask = mask.reshape(1, Mh, 1, Mw, 1)
            mask = mask.expand(-1, -1, window_size, -1, window_size)
            mask = mask.reshape(1, -1)

        # record the corresponding coordinates of visible patches
        coords_h = torch.arange(H, device=x.device)
        coords_w = torch.arange(W, device=x.device)
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w]), dim=-1)  # H W 2
        coords = coords.reshape(1, H * W, 2)

        # for convenient, first divide the image into local windows
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, N, C)
        mask = mask.view(1, H // window_size, window_size, W // window_size,
                         window_size)
        mask = mask.permute(0, 1, 3, 2, 4).reshape(1, N)
        coords = coords.view(1, H // window_size, window_size,
                             W // window_size, window_size, 2)
        coords = coords.permute(0, 1, 3, 2, 4, 5).reshape(1, N, 2)

        # mask out patches
        vis_mask = ~mask  # ~mask means visible
        x_vis = x[vis_mask.expand(B, -1)].reshape(B, -1, C)
        coords = coords[vis_mask].reshape(1, -1, 2)  # 1 N_vis 2

        # transformer forward
        for layer in self.layers:
            x_vis, coords, vis_mask = layer(x_vis, coords, vis_mask)
        x_vis = self.norm(x_vis)
        return x_vis

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # generate random mask: B x Token^2，ids_restore
        mask, ids_restore = self.random_masking(x, mask_ratio)
        latent = self.forward_features(x, mask.bool())

        return latent, mask, ids_restore
