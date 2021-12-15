model = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[0, 1, 2, 3, 4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=-1),
    head=dict(
        type='MultiClsHead',
        pool_type='specified',
        in_indices=[0, 1, 2, 3, 4],
        with_last_layer_unpool=False,
        backbone='resnet50',
        norm_cfg=dict(type='SyncBN', momentum=0.1, affine=False),
        num_classes=1000))
