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
    imgs_per_gpu=128,
    drop_last=False,
    workers_per_gpu=32,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline))

# model
model = dict(backbone=dict(init_cfg=dict(prefix='backbone.')))

# optimizer
optimizer = dict(
    lr=1e-3 * 1024 / 256,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.003697205891018715),
        'cls_token': dict(weight_decay=0.003697205891018715),
        'patch_embed': dict(lr_mult=0.003697205891018715),
        '\\.0\\.': dict(lr_mult=0.005688009063105715),
        '\\.1\\.': dict(lr_mult=0.008750783174008792),
        '\\.2\\.': dict(lr_mult=0.013462743344628911),
        '\\.3\\.': dict(lr_mult=0.02071191283789063),
        '\\.4\\.': dict(lr_mult=0.03186448128906251),
        '\\.5\\.': dict(lr_mult=0.049022278906250015),
        '\\.6\\.': dict(lr_mult=0.07541889062500001),
        '\\.7\\.': dict(lr_mult=0.11602906250000002),
        '\\.8\\.': dict(lr_mult=0.17850625000000003),
        '\\.9\\.': dict(lr_mult=0.274625),
        '\\.10\\.': dict(lr_mult=0.42250000000000004),
        '\\.11\\.': dict(lr_mult=0.65),
        'head': dict(lr_mult=1.0)
    })

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
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
