_base_ = [
    '../../_base_/models/resnet50.py',
    '../../_base_/datasets/imagenet.py',
    '../../_base_/schedules/sgd_steplr-100e.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(backbone=dict(norm_cfg=dict(type='SyncBN')))

# dataset settings
data = dict(
    samples_per_gpu=64,  # total 64x4=256
    train=dict(
        data_source=dict(ann_file='data/imagenet/meta/train_10pct.txt')))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    paramwise_options={'\\Ahead.': dict(lr_mult=1)})

# learning policy
lr_config = dict(policy='step', step=[12, 16], gamma=0.2)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
