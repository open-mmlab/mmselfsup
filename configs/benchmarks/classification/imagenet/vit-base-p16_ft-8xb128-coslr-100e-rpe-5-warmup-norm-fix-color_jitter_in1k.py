_base_ = 'vit-base-p16_ft-8xb128-coslr-100e-rpe-5-warmup_in1k.py'

# dataset
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5))
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    samples_per_gpu=128)

# optimizer
optimizer = dict(lr=8e-3)

# model
model = dict(backbone=dict(init_values=0.1))
