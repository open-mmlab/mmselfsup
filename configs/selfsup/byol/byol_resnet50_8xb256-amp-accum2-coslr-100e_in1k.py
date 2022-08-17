_base_ = 'byol_resnet50_8xb256-amp-accum2-coslr-200e_in1k.py'

# optimizer
optimizer = dict(lr=7.2)
optim_wrapper = dict(optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=90, by_epoch=True, begin=10, end=100)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100)
