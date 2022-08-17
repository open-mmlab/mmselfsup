# model settings
model = dict(
    type='BEiT',
    backbone=dict(
        type='BEiTViT',
        arch='deit-b',
        patch_size=16,
        beit_style=True,
        layer_scale_init_value=0.1,
    ),
    neck=dict(
        type='BEiTNeck',
        num_classes=8192,
        embed_dims=768,
    ),
    head=dict(
        type='BEiTHead',
        tokenizer_path='beit_ckpt/dalle_encoder.pth',
        loss=dict(type='BEiTLoss')),
    data_preprocessor=dict(
        type='mmselfsup.CAEDataPreprocessor',
        mean=[124, 117, 104],
        std=[59, 58, 58],
        bgr_to_rgb=True))
