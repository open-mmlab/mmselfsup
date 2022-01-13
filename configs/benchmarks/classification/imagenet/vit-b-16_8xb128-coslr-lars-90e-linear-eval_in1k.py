_base_ = 'vit-b-16_8xb128-coslr-100e-linear-eval_in1k.py'

# dataset
data = dict(imgs_per_gpu=2048, workers_per_gpu=8)

# optimizer
optimizer = dict(
    type='LARS', lr=0.1 * 16384 / 256, weight_decay=0.0, momentum=0.9)

# learning policy
lr_config = dict(min_lr=0.0)

# runtime
log_config = dict(
    interval=25, hooks=[
        dict(type='TextLoggerHook'),
    ])
