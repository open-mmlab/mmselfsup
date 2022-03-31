# model settings
model = dict(
    type='SimMIM',
    backbone=dict(
        type='SimMIMSwinTransformer',
        arch='B',
        img_size=192,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(type='SimMIMNeck', in_channels=128 * 2**3, encoder_stride=32),
    head=dict(type='SimMIMHead', patch_size=4, encoder_in_channels=3))
