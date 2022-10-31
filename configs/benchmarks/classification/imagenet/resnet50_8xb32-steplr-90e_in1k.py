_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_steplr-100e.py',
    'mmcls::_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(norm_cfg=dict(type='SyncBN')))

# learning rate scheduler
param_scheduler = [
    dict(
        type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)
