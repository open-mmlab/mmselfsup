# model settings
model = dict(
    type='BEiT',
    backbone=dict(
        type='BEiTViT',
        arch='base',
        patch_size=16,
        out_indices=[-4, -1],
        drop_path_rate=0.1,
        final_norm=False,
        beit_style=True,
        layer_scale_init_value=0.1,
    ),
    neck=dict(
        type='BEiTV2Neck',
        early_layers=9,
        num_classes=8192,
        embed_dims=768,
        arch='base',
        shared_lm_head=True,
    ),
    head=dict(
        type='BEiTHead',
        tokenizer_type='vqkd',
        tokenizer_path='beit_ckpt/vqkd_encoder.pth',
        loss=dict(type='BEiTLoss')),
    data_preprocessor=dict(
        type='mmselfsup.BEiTv2DataPreprocessor',
        mean=[124, 117, 104],
        std=[59, 58, 58],
        bgr_to_rgb=True))
