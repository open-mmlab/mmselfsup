# optimizer
optimizer = dict(type='SGD', lr=0.3, momentum=0.9, weight_decay=1e-6)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
