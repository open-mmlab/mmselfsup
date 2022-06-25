# model settings
model = dict(
    type='RotationPred',
    data_preprocessor=dict(
        type='mmselfsup.RotationPredDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    head=dict(
        type='ClsHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        with_avg_pool=True,
        in_channels=2048,
        num_classes=4))
