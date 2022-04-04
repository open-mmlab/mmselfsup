_base_ = [
    '../_base_/models/mocov2.py',
    '../_base_/datasets/imagenet_mocov2.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='MoCo',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNeSt',
        depth=50,
        in_channels=3,
        out_indices=[4],
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='MoCoV2Neck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))


# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
   