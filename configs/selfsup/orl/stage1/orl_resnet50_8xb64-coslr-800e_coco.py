_base_ = [
    '../../_base_/models/byol.py',
    '../../_base_/datasets/coco_orl.py',
    '../../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../../_base_/default_runtime.py',
]

# model settings
model = dict(
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=False,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=False,
            with_avg_pool=False)))

update_interval = 1  # interval for accumulate gradient
# Amp optimizer
optimizer = dict(type='SGD', lr=0.4, weight_decay=0.0001, momentum=0.9)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    accumulative_counts=update_interval,
)
warmup_epochs = 4
total_epochs = 800
# learning policy
param_scheduler = [
    # warmup
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        end=warmup_epochs,
        # Update the learning rate after every iters.
        convert_to_iter_based=True),
    # ConsineAnnealingLR/StepLR/..
    dict(
        type='CosineAnnealingLR',
        eta_min=0.,
        T_max=total_epochs,
        by_epoch=True,
        begin=warmup_epochs,
        end=total_epochs)
]

# runtime settings
default_hooks = dict(checkpoint=dict(interval=100))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs)
