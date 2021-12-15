# model settings
model = dict(
    type='DeepCluster',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(type='AvgPool2dNeck'),
    head=dict(
        type='ClsHead',
        with_avg_pool=False,  # already has avgpool in the neck
        in_channels=2048,
        num_classes=10000))
