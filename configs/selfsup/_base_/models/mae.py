# model settings
model = dict(
    type='MAE',
    backbone=dict(
        type='Vit',
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        pretrain=True,
        embed_dim=768),
    neck=dict(
        type='VitDecoderNeck',  # MAE decoder
        embed_dim=384,
        depth=4,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True),
    head=dict(type='MSEHead', embed_dim=384))
