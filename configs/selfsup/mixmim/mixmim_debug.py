_base_ = [
    '../_base_/models/mixmim.py',
    '../_base_/datasets/imagenet_mixmim.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]


train_dataloader = dict(batch_size=2, num_workers=1)

# optimizer wrapper
optimizer = dict(
    type='AdamW', lr=1.5e-4 * 4096 / 256, betas=(0.9, 0.95), weight_decay=0.05)  # 4096 = 8GPU * 512batchsize
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0,  # √
        by_epoch=True,
        begin=0,
        end=40,  # √
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=560,
        by_epoch=True,
        begin=40,
        end=600,  # √
        convert_to_iter_based=True)
]


train_cfg = dict(max_epochs=600)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)  # √
