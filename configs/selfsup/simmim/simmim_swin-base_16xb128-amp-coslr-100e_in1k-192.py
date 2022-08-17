_base_ = [
    '../_base_/models/simmim_swin-base.py',
    '../_base_/datasets/imagenet_simmim.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset 16 GPUs x 128
train_dataloader = dict(batch_size=128, num_workers=16)

# optimizer wrapper
optimizer = dict(
    type='AdamW', lr=2e-4 * 2048 / 512, betas=(0.9, 0.999), eps=1e-8)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6 / 2e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=90,
        eta_min=1e-5 * 2048 / 512,
        by_epoch=True,
        begin=10,
        end=100,
        convert_to_iter_based=True)
]

# schedule
train_cfg = dict(max_epochs=100)

# runtime
default_hooks = dict(logger=dict(type='LoggerHook', interval=100))
