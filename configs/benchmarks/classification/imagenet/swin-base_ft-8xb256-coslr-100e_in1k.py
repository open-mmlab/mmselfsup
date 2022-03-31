_base_ = [
    '../_base_/models/swin-base.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomAug',
        input_size=192,
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
    dict(type='Resize', size=219, interpolation=3),
    dict(type='CenterCrop', size=192),
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
