custom_imports = dict(imports=['models'], allow_failed_imports=False)

# model settings
img_size = 224
patch_size = 4

model = dict(
    type='GreenMIM',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='GreenMIMSwinTransformer',
        arch='B',
        img_size=img_size,
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
        depths=[2, 2, 18, 2],
        patch_size=patch_size,
        decoder_depth=1,
        drop_path_rate=0.0,
        stage_cfgs=dict(block_cfgs=dict(window_size=7))),
    neck=dict(
        type='GreenMIMNeck',
        in_channels=3,
        encoder_stride=32,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=128),
    head=dict(
        type='GreenMIMHead',
        patch_size=patch_size,
        norm_pix_loss=False,
        loss=dict(type='MAEReconstructionLoss')))
