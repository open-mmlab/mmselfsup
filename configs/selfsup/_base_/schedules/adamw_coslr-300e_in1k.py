# optimizer
optimizer = dict(type='AdamW', lr=6e-4, weight_decay=0.1)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

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
runner = dict(type='EpochBasedRunner', max_epochs=300)
