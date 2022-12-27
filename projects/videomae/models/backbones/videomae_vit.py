# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.utils import to_2tuple
from torch import Tensor, nn

from mmselfsup.models.utils import build_1d_sincos_position_embedding
from mmselfsup.registry import MODELS
from mmselfsup.utils import ConfigType, OptConfigType


class Attention(BaseModule):
    """Multi-head Self-attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads

        self.scale = qk_scale or head_embed_dims**-0.5

        if qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def _init_qv_bias(self) -> None:
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the attention block, same size as inputs.
        """
        B, N, C = x.shape

        if hasattr(self, 'q_bias'):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VideoMAEBlock(BaseModule):
    """The basic block in the Vision Transformer.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        init_values (float): Value to init the multiplier of the
            residual branch. Defaults to 0.
        act_cfg (dict or ConfigDict): Config for activation layer in FFN.
            Defaults to `dict(type='GELU')`.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 init_values: float = 0.0,
                 act_cfg: ConfigType = dict(type='GELU'),
                 norm_cfg: ConfigType = dict(type='LN', eps=1e-6),
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate)

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False)

        self._init_gammas(init_values, embed_dims)

    def _init_gammas(self, init_values: float, dim: int) -> None:
        if type(init_values) == float and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the transformer block, same size as inputs.
        """
        if hasattr(self, 'gamma_1'):
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODELS.register_module()
class VideoMAEViT(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage. An
    impl of `VideoMAE: Masked Autoencoders are Data-Efficient Learners for
    Self-Supervised Video Pre-Training <https://arxiv.org/pdf/2203.12602.pdf>`_

    Args:
        img_size (int or tuple): Size of input image.
            Defaults to 224.
        patch_size (int): Spatial size of one patch. Defaults to 16.
        in_channels (int): The number of channels of he input.
            Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        depth (int): number of blocks in the transformer.
            Defaults to 12.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        norm_cfg (dict or Configdict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        init_values (float): Value to init the multiplier of the residual
            branch. Defaults to 0.
        use_learnable_pos_emb (bool): If True, use learnable positional
            embedding, othersize use sinusoid encoding. Defaults to False.
        num_frames (int): Number of frames in the video. Defaults to 16.
        tubelet_size (int): Temporal size of one patch. Defaults to 2.
        use_mean_pooling (bool): If True, take the mean pooling over all
            positions. Defaults to True.
        pretrained (str, optional): Name of pretrained model. Default: None.
        mask_rato (float): The ratio of masked tokens. Defaults to 0.75.
        mask_type (str): The type of masked tokens.
            Defaults to 'random'. choices=['random', 'tube']
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 3,
            embed_dims: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: int = 4.,
            qkv_bias: bool = True,
            qk_scale: int = None,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_cfg: ConfigType = dict(type='LN', eps=1e-6),
            init_values: int = 0.,
            use_learnable_pos_emb: bool = False,
            num_frames: int = 16,
            tubelet_size: int = 2,
            #  use_mean_pooling: int = True,
            pretrained: Optional[str] = None,
            mask_ratio: float = 0.75,
            mask_type: str = 'random',
            init_cfg: Optional[Union[Dict, List[Dict]]] = [
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ],
            **kwargs) -> None:

        if pretrained:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        patch_size = to_2tuple(patch_size)
        img_size = to_2tuple(img_size)

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv3d',
            kernel_size=(tubelet_size, ) + patch_size,
            stride=(tubelet_size, ) + patch_size,
            padding=(0, 0, 0),
            dilation=(1, 1, 1))

        num_patches = (img_size[1] // patch_size[1]) * \
                      (img_size[0] // patch_size[0]) * \
                      (num_frames // tubelet_size)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dims))
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        else:
            # sine-cosine positional embeddings is on the way
            pos_embed = build_1d_sincos_position_embedding(
                num_patches, embed_dims)
            self.register_buffer('pos_embed', pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = ModuleList([
            VideoMAEBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg,
                init_values=init_values) for i in range(depth)
        ])

        # if use_mean_pooling:
        #     self.norm = nn.Identity()
        #     self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        # else:
        #     self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        #     self.fc_norm = None

        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def video_masking(self, x: torch.Tensor, mask_ratio: float,
                      mask_type: str) -> torch.Tensor:
        """Mask the video feature.

        Args:
            x (torch.Tensor): The video feature.
            mask_ratio (float): The ratio of masked tokens.
            mask_type (str): The type of masked tokens.
                choices=['random', 'tube']
        Returns:
            torch.Tensor: The masked video feature.
        """
        assert mask_type in ['random', 'tube'], \
            f"mask_type must be one of ['random', 'tube'], but got {mask_type}"
        if mask_type == 'random':
            return self.random_masking(x, mask_ratio)
        else:
            return self.tube_masking(x, mask_ratio)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
        Returns:
            Tensor: The feature of the input
                samples extracted by the backbone.
        """
        x = self.patch_embed(x)[0]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        B, _, C = x.shape
        assert mask is not None

        # use ~mask to indicate the visible tokens
        x = x[~mask].reshape(B, -1, C)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x
