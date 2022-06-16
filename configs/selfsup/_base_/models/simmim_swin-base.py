# model settings
model = dict(
    type='SimMIM',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='SimMIMSwinTransformer',
        arch='B',
        img_size=192,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(type='SimMIMNeck', in_channels=128 * 2**3, encoder_stride=32),
    head=dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='SimMIMReconstructionLoss', encoder_in_channels=3)))
