# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmcls.models.backbones.vision_transformer import \
    TransformerEncoderLayer as _TransformerEncoderLayer
from mmcls.models.utils import MultiheadAttention as _MultiheadAttention
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule
from torch.nn import functional as F


class PromptMultiheadAttention(_MultiheadAttention):
    """Prompt Multihead Attention for MILAN.

    This module is specific for the prompt encoder in MILAN. It will not update
    the visible tokens from the encoder.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        return_attention (bool): If True, return the attention map, computed by
            the cross attention between the class token and all other tokens.
            Defaults to False.
        init_cfg (Union[List[dict], dict], optional): The Config for
            initialization. Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 input_dims: Optional[int] = None,
                 attn_drop: float = 0,
                 proj_drop: float = 0,
                 dropout_layer: dict = dict(type='Dropout', drop_prob=0.),
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 proj_bias: bool = True,
                 v_shortcut: bool = False,
                 use_layer_scale: bool = False,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(embed_dims, num_heads, input_dims, attn_drop,
                         proj_drop, dropout_layer, qkv_bias, qk_scale,
                         proj_bias, v_shortcut, use_layer_scale, init_cfg)
        # no longer need qkv
        del self.qkv

        # to project the mask tokens
        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        # to project al the tokens
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)

    def forward(self, x: torch.Tensor, visible_tokens: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """Forward function for `PromptMultiheadAttention`.

        Args:
            x (torch.Tensor): Mask token features with shape N x L_m x C.
            visible_tokens (torch.Tensor): The visible tokens features from
                encoder with shape N x L_v x C.
            ids_restore (torch.Tensor): The ids of all tokens in the original
                image with shape N x L.

        Returns:
            torch Tensor: Output features with shape N x L x C.
        """
        x_ = torch.cat([visible_tokens[:, 1:, :], x], dim=1)
        assert x_.shape[1] == ids_restore.shape[1]
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        x_ = torch.cat([visible_tokens[:, :1, :], x_], dim=1)

        # full sequence shape
        B, _, _ = x_.shape
        q = self.q(x).reshape(B, x.shape[1], self.num_heads,
                              self.head_dims).permute(0, 2, 1, 3)
        kv = self.kv(x_).reshape(B, x_.shape[1], 2, self.num_heads,
                                 self.head_dims).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, x.shape[1], self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))
        return x


class PromptTransformerEncoderLayer(_TransformerEncoderLayer):
    """Prompt Transformer Encoder Layer for MILAN.

    This module is specific for the prompt encoder in MILAN. It will not update
    the visible tokens from the encoder.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): Enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels=int,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.attn = PromptMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias)

    def forward(self, x: torch.Tensor, visible_tokens: torch.Tensor,
                ids_restore: torch.Tensor) -> torch.Tensor:
        """Forward function for `PromptMultiheadAttention`.

        Args:
            x (torch.Tensor): Mask token features with shape N x L_m x C.
            visible_tokens (torch.Tensor): The visible tokens features from
                encoder with shape N x L_v x C.
            ids_restore (torch.Tensor): The ids of all tokens in the original
                image with shape N x L.

        Returns:
            torch Tensor: Output features with shape N x L x C.
        """
        x = x + self.attn(self.norm1(x), visible_tokens, ids_restore)
        x = self.ffn(self.norm2(x), identity=x)
        return x


class MultiheadAttention(_MultiheadAttention):
    """Multi-head Attention Module.

    This module rewrite the MultiheadAttention by replacing qkv bias with
    customized qkv bias, in addition to removing the drop path layer.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 input_dims: int = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 proj_bias: bool = True,
                 init_cfg: dict = None) -> None:
        super(MultiheadAttention, self).__init__(
            embed_dims,
            num_heads=num_heads,
            input_dims=input_dims,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias,
            init_cfg=init_cfg)

        self.qkv_bias = qkv_bias

        if not self.qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(
            self.input_dims, embed_dims * 3, bias=self.qkv_bias)

    def _init_qv_bias(self):
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        # qkv bias is different from that in mmcls
        B, N, _ = x.shape

        if not self.qkv_bias:
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MultiheadAttentionWithRPE(MultiheadAttention):
    """Multi-head Attention Module.

    This module rewrite the MultiheadAttention in MMSelfSup by adding the
    relative position bias.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        window_size (int): The window size of the relative position bias.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 window_size: int,
                 input_dims: int = None,
                 attn_drop: float = 0,
                 proj_drop: float = 0,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 proj_bias: bool = True,
                 init_cfg: dict = None) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            input_dims=input_dims,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias,
            init_cfg=init_cfg)

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        assert isinstance(window_size, Sequence)
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        # relative_position_bias_table shape is (2*Wh-1 * 2*Ww-1 + 3, nH)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for
        # each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # coords shape is (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # coords_flatten shape is (2, Wh*Ww)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :])
        # relative_coords shape is (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, ) * 2,
            dtype=relative_coords.dtype)

        # relative_position_index shape is (Wh*Ww, Wh*Ww)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index',
                             relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        B, N, _ = x.shape
        qkv = F.linear(
            x, weight=self.qkv.weight,
            bias=qkv_bias).reshape(B, N, 3, self.num_heads,
                                   self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[
                    self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(_TransformerEncoderLayer):
    """Implements one encoder layer in Vision Transformer.

    This module is the rewritten version of the TransformerEncoderLayer in
    MMClassification by adding the gamma and relative position bias in
    Attention module.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_values (float): The init values of gamma. Defaults to 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 window_size: int = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 init_values: float = 0.0,
                 init_cfg: dict = None) -> None:
        super(TransformerEncoderLayer, self).__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            num_fcs=num_fcs,
            qkv_bias=qkv_bias,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg)
        self.embed_dims = embed_dims

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        if window_size is None:
            # attention without relative position bias
            self.attn = MultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                qkv_bias=qkv_bias)
        else:
            # attention with relative position bias
            self.attn = MultiheadAttentionWithRPE(
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=window_size,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                qkv_bias=qkv_bias)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=None,
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class CAETransformerRegressorLayer(BaseModule):
    """Transformer layer for the regressor of CAE.

    This module is different from conventional transformer encoder layer, for
    its queries are the masked tokens, but its keys and values are the
    concatenation of the masked and unmasked tokens.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): The number of heads in multi-head attention.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): The dropout rate. Defaults to 0.0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        init_values (float): The init values of gamma. Defaults to 0.0.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        num_fcs: int = 2,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        init_values: float = 0.0,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN', eps=1e-6)
    ) -> None:
        super().__init__()

        # NOTE: cross attention
        _, self.norm1_q_cross = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        _, self.norm1_k_cross = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        _, self.norm1_v_cross = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        _, self.norm2_cross = build_norm_layer(norm_cfg, embed_dims, postfix=2)
        self.cross_attn = CrossMultiheadAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=None,
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if init_values > 0:
            self.gamma_1_cross = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
            self.gamma_2_cross = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
        else:
            self.gamma_1_cross = nn.Parameter(
                torch.ones((embed_dims)), requires_grad=False)
            self.gamma_2_cross = nn.Parameter(
                torch.ones((embed_dims)), requires_grad=False)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor,
                pos_q: torch.Tensor, pos_k: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = x_q + self.drop_path(self.gamma_1_cross * self.cross_attn(
            self.norm1_q_cross(x_q + pos_q),
            k=self.norm1_k_cross(x_kv + pos_k),
            v=self.norm1_v_cross(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.ffn(x))

        return x


class CrossMultiheadAttention(BaseModule):
    """Cross attention between queries and the union of keys and values.

    This module is different from ``MultiheadAttention``, for the attention
    is computed between queries and the union of keys and values.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = False,
                 qk_scale: float = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(embed_dims, embed_dims, bias=False)
        self.k = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v = nn.Linear(embed_dims, embed_dims, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
                x: torch.Tensor,
                k: torch.Tensor = None,
                v: torch.Tensor = None) -> None:
        """Forward function."""
        B, N, _ = x.shape

        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(
            input=x, weight=self.q.weight, bias=q_bias)  # (B, N_q, dim)
        k = F.linear(
            input=k, weight=self.k.weight, bias=k_bias)  # (B, N_k, dim)
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)

        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape(B, N_k, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape(B, N_v, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
