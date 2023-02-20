_base_ = [
    'mmcls::_base_/models/convnext/convnext-small.py',
    'mmcls::_base_/datasets/imagenet_bs64_swin_224.py',
    'mmcls::_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'ImageNet'
data_root = 'sproject:s3://openmmlab/datasets/classification/imagenet/'

data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=236,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', shuffle=True),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root, pipeline=test_pipeline))
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmselfsup.CosineEMA',
        momentum=0.99,
        end_momentum=0.999,
        priority='ABOVE_NORMAL')
]

# schedule settings
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=3.2e-3, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=None,
    paramwise_cfg=dict(
        norm_decay_mult=0.0, bias_decay_mult=0.0, flat_decay_mult=0.0))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=280,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=20,
        end=200)
]
train_cfg = dict(by_epoch=True, max_epochs=200)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    # only keeps the latest 2 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
