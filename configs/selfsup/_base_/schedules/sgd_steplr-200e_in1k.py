# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=1e-4, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[120, 160], gamma=0.1)
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
