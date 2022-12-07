_base_ = '../../../benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'  # noqa: E501

# optimizer
optimizer = dict(type='mmselfsup.LARS', lr=3.2, weight_decay=0.0, momentum=0.9)
optim_wrapper = dict(
    type='AmpOptimWrapper', optimizer=optimizer, _delete_=True)

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=100)

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
        type='CosineAnnealingLR',
        T_max=90,
        by_epoch=True,
        begin=10,
        end=100,
        eta_min=0.0,
        convert_to_iter_based=True)
]
