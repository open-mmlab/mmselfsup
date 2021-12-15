_base_ = [
    '../_base_/models/simclr.py',
    '../_base_/datasets/imagenet_simclr.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(
    type='LARS',
    lr=0.3,
    momentum=0.9,
    weight_decay=1e-6,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)':
        dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True)
    })

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,
    warmup_by_epoch=True)

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
