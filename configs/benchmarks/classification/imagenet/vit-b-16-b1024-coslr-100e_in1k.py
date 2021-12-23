_base_ = [
    '../_base_/models/vit_base_16.py',
    '../_base_/datasets/mae_imagenet.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(
    lr=1e-3 * 1024 / 256,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'patch_embed': dict(lr_mult=0.023757264018058777),
        '\\.0\\.': dict(lr_mult=0.03167635202407837),
        '\\.1\\.': dict(lr_mult=0.04223513603210449),
        '\\.2\\.': dict(lr_mult=0.056313514709472656),
        '\\.3\\.': dict(lr_mult=0.07508468627929688),
        '\\.4\\.': dict(lr_mult=0.1001129150390625),
        '\\.5\\.': dict(lr_mult=0.13348388671875),
        '\\.6\\.': dict(lr_mult=0.177978515625),
        '\\.7\\.': dict(lr_mult=0.2373046875),
        '\\.8\\.': dict(lr_mult=0.31640625),
        '\\.9\\.': dict(lr_mult=0.421875),
        '\\.10\\.': dict(lr_mult=0.5625),
        '\\.11\\.': dict(lr_mult=0.75),
        'head': dict(lr_mult=1.0)
    })

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-4,
    warmup_by_epoch=True)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    out_dir='/mnt/lustre/liuyuan1.vendor/ckpt/mae_ft')

persistent_workers = True
runner = dict(max_epochs=100)

log_config = dict(
    interval=1, hooks=[
        dict(type='TextLoggerHook'),
    ])

data = dict(imgs_per_gpu=128)

model = dict(backbone=dict(init_cfg=dict(prefix='backbone.')))
