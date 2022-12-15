_base_ = [
    '../_base_/models/mixmim.py',
    '../_base_/datasets/imagenet_mixmim.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# optimizer wrapper
optimizer = dict(
    type='AdamW',
    lr=1.5e-4 *
    (2 * 8 * 128 / 256),  # total_lr = base_lr*num_gpus*base_bs/256 = 1.2e-3
    betas=(0.9, 0.95),
    weight_decay=0.05)  # 2 node * 8 gpu * 128 batchsize
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={
        'ln': dict(decay_mult=0.0),
        'bias': dict(decay_mult=0.0)
    }))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

train_cfg = dict(max_epochs=300)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
