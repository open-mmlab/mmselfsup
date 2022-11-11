# model settings
model = dict(
    type='BEiT',
    backbone=dict(
        type='BEiTViT',
        arch='base',
        patch_size=16,
        drop_path_rate=0.1,
        final_norm=True,
        layer_scale_init_value=0.1,
    ),
    neck=None,
    head=dict(
        type='BEiTV1Head',
        embed_dims=768,
        num_embed=8192,
        loss=dict(type='BEiTLoss')),
    target_generator=dict(
        type='DALL-E',
        init_cfg=dict(
            type='Pretrained', checkpoint='beit_ckpt/dalle_encoder.pth')))
