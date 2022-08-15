_base_ = 'mocov3_vit-base-p16_16xb256-fp16-coslr-300e_in1k-224.py'

# optimizer
optimizer = dict(type='AdamW', lr=2.4e-3, weight_decay=0.1)
optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic', optimizer=optimizer)
find_unused_parameters = True

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

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=800)
# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))
