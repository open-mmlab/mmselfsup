# model settings
model = dict(
    type='VideoMAE',
    data_preprocessor=dict(
        type='VideoMAEDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    backbone=dict(
        type='VideoMAEViT',
        img_size=224,
        embed_dims=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6),
        patch_size=16,
        mask_ratio=0.9),
    neck=dict(
        type='VideoMAEPretrainDecoder',
        img_size=224,
        num_frames=16,
        num_classes=1536,
        num_heads=3,
        input_dims=384,
        embed_dims=192,
        patch_size=16,
    ),
    head=dict(
        type='VideoMAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=[
        dict(type='Xavier', distribution='uniform', layer='Linear'),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ])
