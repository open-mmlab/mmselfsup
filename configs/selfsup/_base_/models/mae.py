# model settings
model = dict(
    type='MAE',
    backbone=dict(
        type='MAEPretrainViT',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        mask_ratio=0.75),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(type='MAEPretrainHead', norm_pix_loss=False, patch_size=16),
    fp16_enabled=True)
