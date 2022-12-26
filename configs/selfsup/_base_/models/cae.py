# model settings
model = dict(
    type='CAE',
    backbone=dict(
        type='CAEViT',
        arch='b',
        patch_size=16,
        init_values=0.1,
        qkv_bias=False),
    neck=dict(
        type='CAENeck',
        patch_size=16,
        embed_dims=768,
        num_heads=12,
        regressor_depth=4,
        decoder_depth=4,
        mlp_ratio=4,
        init_values=0.1,
    ),
    head=dict(type='CAEHead', loss=dict(type='CAELoss', lambd=2)),
    target_generator=dict(
        type='DALL-E',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/dalle_encoder.pth',  # noqa: E501
        )),
    data_preprocessor=dict(
        type='mmselfsup.CAEDataPreprocessor',
        mean=[124, 117, 104],
        std=[59, 58, 58],
        bgr_to_rgb=True),
    base_momentum=0.0)
