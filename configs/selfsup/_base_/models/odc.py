# model settings
model = dict(
    type='ODC',
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='ODCNeck',
        in_channels=2048,
        hid_channels=512,
        out_channels=256,
        with_avg_pool=True),
    head=dict(
        type='ClsHead',
        with_avg_pool=False,
        in_channels=256,
        num_classes=10000),
    loss=dict(type='mmcls.CrossEntropyLoss'),
    memory_bank=dict(
        type='ODCMemory',
        length=1281167,
        feat_dim=256,
        momentum=0.5,
        num_classes=10000,
        min_cluster=20,
        debug=False))
