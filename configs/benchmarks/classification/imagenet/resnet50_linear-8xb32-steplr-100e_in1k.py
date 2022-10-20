_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_steplr-100e.py',
    'mmcls::_base_/default_runtime.py',
]
# MoCo v1/v2 linear evaluation setting

model = dict(backbone=dict(frozen_stages=4))

# optimizer
optimizer = dict(type='SGD', lr=30., momentum=0.9, weight_decay=0.)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
