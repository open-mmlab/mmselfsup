# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='VisionTransformer',
        arch='mocov3-small',  # embed_dim = 384
        img_size=224,
        patch_size=16,
        stop_grad_conv1=True),
    head=dict(
        type='ClsHead',
        in_channels=384,
        num_classes=1000,
        vit_backbone=True,
    ))
