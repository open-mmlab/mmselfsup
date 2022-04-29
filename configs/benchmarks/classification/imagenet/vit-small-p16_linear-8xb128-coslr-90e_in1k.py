_base_ = [
    '../_base_/models/vit-small-p16.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    '../_base_/default_runtime.py',
]
# MoCo v3 linear probing setting

model = dict(backbone=dict(frozen_stages=12, norm_eval=True))

# dataset summary
data = dict(samples_per_gpu=128)  # total 128*8=1024, 8 GPU linear cls

# optimizer
optimizer = dict(type='SGD', lr=12, momentum=0.9, weight_decay=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)

# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
