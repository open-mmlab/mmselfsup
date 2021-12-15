# optimizer
optimizer = dict(type='LARS', lr=4.8, weight_decay=1e-6, momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-4,  # cannot be 0
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)
