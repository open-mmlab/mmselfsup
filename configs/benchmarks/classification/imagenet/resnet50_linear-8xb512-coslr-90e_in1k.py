_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/lars_coslr-90e.py',
    '../_base_/default_runtime.py',
]
# SimSiam linear evaluation setting
# According to SimSiam paper, this setting can also be used to evaluate
# other methods like SimCLR, MoCo, BYOL, SwAV

model = dict(backbone=dict(frozen_stages=4))

# dataset summary
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=8)  # total 512*8=4096, 8GPU linear cls

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
