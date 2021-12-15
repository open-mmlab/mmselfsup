_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_steplr-100e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(norm_cfg=dict(type='SyncBN')))

# learning policy
lr_config = dict(step=[30, 60, 90])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3, out_dir='s3://results')
