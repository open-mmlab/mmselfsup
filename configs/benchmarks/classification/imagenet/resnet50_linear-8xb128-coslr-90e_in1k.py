_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    'mmcls::_base_/default_runtime.py',
]
# MoCoV3 ResNet50 linear probing setting

model = dict(backbone=dict(frozen_stages=4, norm_eval=True))

# dataset summary
train_dataloader = dict(batch_size=128)

# optimizer
optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=90, by_epoch=True, begin=0, end=90)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
