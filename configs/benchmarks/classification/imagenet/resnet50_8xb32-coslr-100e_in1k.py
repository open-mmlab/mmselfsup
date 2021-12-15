_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    '../_base_/default_runtime.py',
]

model = dict(backbone=dict(frozen_stages=4))

# swav setting
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
