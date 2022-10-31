_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    'mmcls::_base_/default_runtime.py',
]
# SwAV and Barlow Twins linear evaluation setting

model = dict(backbone=dict(frozen_stages=4))

# runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
