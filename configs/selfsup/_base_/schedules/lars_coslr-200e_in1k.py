# optimizer
optimizer = dict(type='LARS', lr=4.8, weight_decay=1e-6, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=190,
        by_epoch=False,
        begin=10,
        end=200,
        convert_to_iter_based=True)
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
