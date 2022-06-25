_base_ = [
    '../_base_/models/rotation-pred.py',
    '../_base_/datasets/imagenet_rotation-pred.py',
    '../_base_/schedules/sgd_steplr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(type='MultiStepLR', by_epoch=True, milestones=[30, 50], gamma=0.1)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=70)
