# optimizer
optimizer = dict(type='LARS', lr=4.8, weight_decay=1e-6, momentum=0.9)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=190, by_epoch=True, begin=10, end=200)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
