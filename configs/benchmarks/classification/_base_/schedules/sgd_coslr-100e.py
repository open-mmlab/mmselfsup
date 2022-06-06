# optimizer
optimizer = dict(type='SGD', lr=0.3, momentum=0.9, weight_decay=1e-6)

# learning rate scheduler
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100)
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
