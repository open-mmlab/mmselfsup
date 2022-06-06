# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100)
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
