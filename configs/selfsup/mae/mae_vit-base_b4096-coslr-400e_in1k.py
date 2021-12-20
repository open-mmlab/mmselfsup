_base_ = [
    '../_base_/models/mae.py',
    '../_base_/datasets/imagenet_mae.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(
    lr=1.5e-4 * 4096 / 256,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'mask_token': dict(weight_decay=0.)
    })

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=40,
    warmup_ratio=1e-4,
    warmup_by_epoch=True)

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    out_dir='/mnt/lustre/liuyuan1.vendor/ckpt/mae')

persistent_workers = True
runner = dict(max_epochs=400)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

data = dict(imgs_per_gpu=128)

# dist_params = dict(backend='nccl', port=29500)
