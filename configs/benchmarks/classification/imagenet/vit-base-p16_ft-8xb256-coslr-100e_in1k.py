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
