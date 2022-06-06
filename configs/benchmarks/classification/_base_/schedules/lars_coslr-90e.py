# optimizer
optimizer = dict(type='LARS', lr=1.6, momentum=0.9, weight_decay=0.)

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=90, by_epoch=True, begin=0, end=90)
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=90)
