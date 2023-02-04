_base_ = [
    '../_base_/models/videomae_vit-small-p16.py',
    '../_base_/datasets/k400_videomae.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    'mmselfsup::selfsup/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['models', 'datasets', 'mmaction.datasets.transforms'],
    allow_failed_imports=False)

# dataset 2 * 8 * 64 = 1024
train_dataloader = dict(batch_size=64, num_workers=8)
# optimizer wrapper
optimizer = dict(
    type='AdamW', lr=1.5e-4 * 1024 / 256, betas=(0.9, 0.95), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            # 'ln': dict(decay_mult=0.0),
            # 'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            'decoder_pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.)
        }))

# learning rate scheduler
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
        T_max=760,
        by_epoch=True,
        begin=40,
        end=800,
        convert_to_iter_based=True)
]

train_cfg = dict(max_epochs=800)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
resume = True
