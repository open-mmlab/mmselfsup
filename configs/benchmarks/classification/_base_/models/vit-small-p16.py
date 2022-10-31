# model settings
model = dict(
    type='ImageClassifier',
    data_preprocessor=dict(
        num_classes=1000,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    backbone=dict(
        type='mmselfsup.MoCoV3ViT',
        arch='mocov3-small',  # embed_dim = 384
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='Normal', std=0.01, layer='Linear'),
    ))
