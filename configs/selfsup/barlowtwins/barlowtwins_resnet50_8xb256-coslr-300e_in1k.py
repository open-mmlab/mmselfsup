_base_ = [
    '../_base_/models/barlowtwins.py',
    '../_base_/datasets/imagenet_byol.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

data = dict(samples_per_gpu=256)

# optimizer
optimizer = dict(
    type='LARS',
    lr=1.6,
    momentum=0.9,
    weight_decay=1e-6,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)':
        dict(weight_decay=0, lr_mult=0.024, lars_exclude=True),
        'bias':
        dict(weight_decay=0, lr_mult=0.024, lars_exclude=True),
        # bn layer in ResNet block downsample module
        'downsample.1':
        dict(weight_decay=0, lr_mult=0.024, lars_exclude=True),
    })

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0.0016,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1.6e-4,  # cannot be 0
    warmup_by_epoch=True)

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)

runner = dict(type='EpochBasedRunner', max_epochs=300)
