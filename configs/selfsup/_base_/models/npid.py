# model settings
model = dict(
    type='NPID',
    neg_num=65536,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='LinearNeck',
        in_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.07),
    memory_bank=dict(
        type='SimpleMemory', length=1281167, feat_dim=128, momentum=0.5))
