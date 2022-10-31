_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet.py',
    '../../_base_/schedules/sgd_steplr-100e.py',
    'mmcls::_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(norm_cfg=dict(type='SyncBN')))

# dataset settings
train_dataloader = dict(
    batch_size=64,  # total 64x4=256
    dataset=dict(ann_file='meta/train_1percent.txt'))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=1.)}))

# learning rate scheduler
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[12, 16], gamma=0.2)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20)
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=10))
