_base_ = [
    '../_base_/models/maskfeat_vit-base-p16.py',
    '../_base_/datasets/imagenet_maskfeat.py',
    '../_base_/schedules/adamw_coslr-300e_in1k.py',
    '../_base_/default_runtime.py',
]

# optimizer wrapper
optimizer = dict(lr=2e-4 * 8, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=optimizer,
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            # 'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
            # 'cls_token': dict(decay_mult=0.)
        }),
    # commented 'pos_embed' and 'cls_token' to avoid loss stuck situation
    clip_grad=dict(max_norm=0.02))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=30,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=270,
        by_epoch=True,
        begin=30,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
# pre-train for 300 epochs
train_cfg = dict(max_epochs=300)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))
