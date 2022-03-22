_base_ = [
    '../_base_/models/simmim_swin-base.py',
    '../_base_/datasets/imagenet_simmim.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# data
data = dict(samples_per_gpu=128)

# optimizer
optimizer = dict(
    lr=2e-4 * 2048 / 512,
    betas=(0.9, 0.999),
    eps=1e-8,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'absolute_pos_embed': dict(weight_decay=0.),
        'relative_position_bias_table': dict(weight_decay=0.0)
    })

# clip gradient
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5 * 2048 / 512,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-6 / 2e-4,
    warmup_by_epoch=True,
    by_epoch=False)

# mixed precision
fp16 = dict(loss_scale='dynamic')

# schedule
runner = dict(max_epochs=100)

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='')
persistent_workers = True
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])
