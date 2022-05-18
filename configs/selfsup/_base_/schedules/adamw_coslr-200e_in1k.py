# optimizer
optimizer = dict(type='AdamW', lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.05)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=False,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
