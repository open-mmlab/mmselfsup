_base_ = [
    '../_base_/models/vit-base-p16_ft.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomAug',
        input_size=224,
        color_jitter=None,
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
    samples_per_gpu=128,
    drop_last=False,
    workers_per_gpu=32,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline))

# model
model = dict(backbone=dict(init_cfg=dict()))

# optimizer
optimizer = dict(
    lr=1e-3 * 1024 / 256,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.)
    },
    constructor='MAEFtOptimizerConstructor',
    layer_decay=0.65)

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-4,
    warmup_by_epoch=True,
    by_epoch=False)

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='')
persistent_workers = True
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])
