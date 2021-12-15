# model settings
model = dict(
    type='RelativeLoc',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='RelativeLocNeck',
        in_channels=2048,
        out_channels=4096,
        with_avg_pool=True),
    head=dict(
        type='ClsHead',
        with_avg_pool=False,
        in_channels=4096,
        num_classes=8,
        init_cfg=[
            dict(type='Normal', std=0.005, layer='Linear'),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]))
