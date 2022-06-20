# optimizer
optimizer = dict(type='AdamW', lr=6e-4, weight_decay=0.1)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

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
        type='CosineAnnealingLR', T_max=260, by_epoch=True, begin=40, end=300)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
