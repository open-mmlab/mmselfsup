_base_ = [
    '../_base_/models/resnet50_multihead.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_steplr-100e.py',
    'mmcls::_base_/default_runtime.py',
]

# lighting params, in order of BGR
EIGVAL = [55.4625, 4.7940, 1.1475]
EIGVEC = [
    [-0.5836, -0.6948, 0.4203],
    [-0.5808, -0.0045, -0.8140],
    [-0.5675, 0.7192, 0.4009],
]

# dataset
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.),
    dict(
        type='Lighting',
        eigval=EIGVAL,
        eigvec=EIGVEC,
    ),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# MoCo v1/v2 linear evaluation setting
model = dict(
    backbone=dict(out_indices=[0, 1, 2, 3], frozen_stages=4),
    head=dict(in_indices=[1, 2, 3, 4]))

# optimizer
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(norm_decay_mult=0.0))

# runtime settings
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

# evaluator
val_evaluator = dict(
    _delete_=True,
    type='MultiHeadEvaluator',
    metrics=dict(
        head1=dict(type='mmcls.Accuracy', topk=(1, 5)),
        head2=dict(type='mmcls.Accuracy', topk=(1, 5)),
        head3=dict(type='mmcls.Accuracy', topk=(1, 5)),
        head4=dict(type='mmcls.Accuracy', topk=(1, 5))))

# epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90)
