_base_ = [
    '../_base_/models/resnet50.py',
    'mmcls::_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/sgd_steplr-100e.py',
    'mmcls::_base_/default_runtime.py',
]

# dataset settings
train_dataloader = dict(batch_size=128)
val_dataloader = dict(batch_size=128)

# model settings
model = dict(data_preprocessor=dict(num_classes=10), head=dict(num_classes=10))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[150, 250], gamma=0.1)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=350)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3))
