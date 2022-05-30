_base_ = ['swin-base_ft-8xb256-coslr-100e_in1k.py']

# model
model = dict(
    backbone=dict(
        img_size=224, stage_cfgs=dict(block_cfgs=dict(window_size=7))))

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
        scale=224,
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
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackSelfSupInputs', algorithm_keys=['gt_label']),
]

data = dict(
    samples_per_gpu=256,
    drop_last=False,
    workers_per_gpu=32,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline))
