_base_ = 'byol_resnet50_8xb32-accum16-coslr-200e_in1k.py'

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
        type='CosineAnnealingLR', T_max=290, by_epoch=True, begin=10, end=300)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
