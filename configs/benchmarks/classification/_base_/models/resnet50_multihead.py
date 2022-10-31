model = dict(
    type='ImageClassifier',
    data_preprocessor=dict(
        num_classes=1000,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[0, 1, 2, 3, 4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=-1),
    head=dict(
        type='mmselfsup.MultiClsHead',
        pool_type='specified',
        in_indices=[0, 1, 2, 3, 4],
        with_last_layer_unpool=False,
        backbone='resnet50',
        norm_cfg=dict(type='SyncBN', momentum=0.1, affine=False),
        num_classes=1000))
