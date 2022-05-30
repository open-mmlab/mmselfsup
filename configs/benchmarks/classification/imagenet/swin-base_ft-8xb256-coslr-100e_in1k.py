_base_ = [
    '../_base_/models/swin-base.py', '../_base_/datasets/imagenet.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    '../_base_/default_runtime.py', '../_base_/datasets/pipelines/rand_aug.py'
]

# dataset
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)
preprocess_cfg = dict(
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

bgr_mean = preprocess_cfg['pixel_mean'][::-1]
bgr_std = preprocess_cfg['pixel_std'][::-1]

# train pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='mmcls.RandomResizedCrop',
        scale=192,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='mmcls.RandAugment',
        policies={{_base_.rand_increasing_policies}},
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='mmcls.RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackSelfSupInputs', algorithm_keys=['gt_label']),
]

# test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='mmcls.ResizeEdge',
        scale=219,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=192),
    dict(type='PackSelfSupInputs', algorithm_keys=['gt_label']),
]

data = dict(
    samples_per_gpu=256,
    drop_last=False,
    workers_per_gpu=32,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline))

# model
model = dict(backbone=dict(init_cfg=dict()))

# optimizer
optimizer = dict(
    lr=1.25e-3 * 2048 / 512,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'absolute_pos_embed': dict(weight_decay=0.),
        'relative_position_bias_table': dict(weight_decay=0.)
    },
    constructor='TransformerFinetuneConstructor',
    model_type='swin',
    layer_decay=0.9)

# clip gradient
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=2.5e-7 * 2048 / 512,
    warmup='linear',
    warmup_iters=20,
    warmup_ratio=2.5e-7 / 1.25e-3,
    warmup_by_epoch=True,
    by_epoch=False)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='')
persistent_workers = True
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])
