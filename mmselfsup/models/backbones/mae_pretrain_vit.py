from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from timm.models.vision_transformer import PatchEmbed, Block

from ..builder import BACKBONES


@BACKBONES.register_module()
class MAEPretrainViT(BaseModule):
    """ViT backbone for MAE pre-training.

    Args:
        img_size (int): Input image size, Defaults to 224.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): embedding dimension. Defaults to 1024.
        depth (int): Depth of transformer. Defaults to 24.
        num_heads (int): Number of attention heads. Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        norm_layer (nn.Module): Normalization layer. Defaults to nn.LayerNorm.
        mask_ratio (float): Mask ratio of total patches. Defaults to 0.75.
        init_cfg (dict, optional): Initialization config dict. Defaults to
            None.

    Some of the code is borrowed from
    `https://github.com/facebookresearch/mae`.

    Example:
        >>> from mmselfsup.models import MAEPretrainViT
        >>> import torch
        >>> self = MAEPretrainViT()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> level_outputs = self.forward(inputs)
        >>> print(tuple(level_outputs.shape))
        (1, 50, 1024)
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=12,
                 num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 mask_ratio=0.75,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.mask_ratio = mask_ratio

    def random_masking(self, x, mask_ratio=0.75):
        """ Generate the mask for MAE Pre-training.

        Args:
            x (torch.tensor): Image with data augmentation applied.
            mask_ratio (float): The mask ratio of total patches.
                Defaults to 0.75.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return (x, mask, ids_restore)
