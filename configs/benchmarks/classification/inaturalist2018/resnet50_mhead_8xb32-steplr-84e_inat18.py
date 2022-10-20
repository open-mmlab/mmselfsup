_base_ = [
    '../_base_/models/resnet50_multihead.py',
    '../_base_/datasets/inaturalist2018.py',
    '../_base_/schedules/sgd_steplr-100e.py',
    'mmcls::_base_/default_runtime.py',
]

# model settings
model = dict(
    backbone=dict(frozen_stages=4),
    head=dict(
        norm_cfg=dict(type='SyncBN', momentum=0.1, affine=False),
        num_classes=8142))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    paramwise_options=dict(norm_decay_mult=0.),
    nesterov=True)

# learning rate scheduler
param_scheduler = [
    dict(
        type='MultiStepLR', by_epoch=True, milestones=[24, 48, 72], gamma=0.1)
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=84)
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
