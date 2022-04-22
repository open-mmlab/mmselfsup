_base_ = [
    '../_base_/models/cae.py',
    '../_base_/datasets/imagenet_cae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset
data = dict(samples_per_gpu=64, workers_per_gpu=8)

# optimizer
optimizer = dict(
    lr=1.5e-3,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.)
    },
    betas=(0.9, 0.999))

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,
    warmup_by_epoch=True,
    by_epoch=False)

# schedule
runner = dict(max_epochs=300)

# clip gradient
optimizer_config = dict(grad_clip=dict(max_norm=3.0))

# mixed precision
fp16 = dict(loss_scale='dynamic')

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=2, out_dir='')
persistent_workers = True
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])

find_unused_parameters = True
