<<<<<<< HEAD
# mmcls:: means we use the default settings from MMClassification
_base_ = 'vit-base-p16_ft-8xb128-coslr-100e_in1k.py'
# maskfeat fine-tuning setting

# model settings
model = dict(
    head=dict(init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=2e-5),
    ]))

# dataset
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
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
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(batch_size=256, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=256, dataset=dict(pipeline=test_pipeline))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader

# optimizer wrapper
optim_wrapper = dict(optimizer=dict(lr=0.008))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        by_epoch=True,
        begin=20,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(max_epochs=100)
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0)
=======
_base_ = [
    '../_base_/models/vit-base-p16_ft.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    '../_base_/default_runtime.py',
]
# maskfeat fine-tuning setting

# dataset
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomAug',
        input_size=224,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
data = dict(
    samples_per_gpu=256,
    drop_last=False,
    workers_per_gpu=32,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline))

# model
model = dict(
    backbone=dict(init_cfg=dict()),
    head=dict(
        type='MaskFeatFinetuneHead',
        num_classes=1000,
        embed_dim=768,
        label_smooth_val=0.1))

# optimizer
optimizer = dict(
    lr=0.002 * 8 / 2,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_options={
        'ln': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
    },
    constructor='TransformerFinetuneConstructor',
    model_type='vit',
    layer_decay=0.65)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=1e-08,
    warmup_by_epoch=True)

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='')
persistent_workers = True
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])
>>>>>>> upstream/master
