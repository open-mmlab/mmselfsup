_base_ = [
    '../_base_/models/maskfeat_vit-base-p16.py',
    '../_base_/datasets/imagenet_maskfeat.py',
    '../_base_/schedules/adamw_coslr-300e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset
data = dict(samples_per_gpu=256, workers_per_gpu=32)

# optimizer
optimizer = dict(lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=0.02))

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=0.0,
    warmup='linear',
    warmup_iters=30,
    warmup_by_epoch=True)

# schedule
runner = dict(max_epochs=300)

# runtime
checkpoint_config = dict(interval=1, max_keep_ckpts=3, out_dir='')
persistent_workers = True
log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])
