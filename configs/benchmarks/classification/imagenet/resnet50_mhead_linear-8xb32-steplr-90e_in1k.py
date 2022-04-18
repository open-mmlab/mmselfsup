_base_ = [
    '../_base_/models/resnet50_multihead.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_steplr-100e.py',
    '../_base_/default_runtime.py',
]

model = dict(backbone=dict(frozen_stages=4))

# dataset settings
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.),
    dict(type='ToTensor'),
    dict(type='Lighting'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    train=dict(pipeline=train_pipeline), val=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    paramwise_options=dict(norm_decay_mult=0.),
    nesterov=True)

# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)
checkpoint_config = dict(interval=10)
