# model settings
model = dict(
    type='ImageClassifier',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    backbone=dict(
        type='mmselfsup.MoCoV3ViT',
        arch='base',  # embed_dim = 768
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='Normal', std=0.01, layer='Linear'),
    ))
