_base_ = [
    '../_base_/models/simsiam.py',
    '../_base_/datasets/imagenet_mocov2.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# set base learning rate
lr = 0.05

# additional hooks
custom_hooks = [
    dict(type='SimSiamHook', priority='HIGH', fix_pred_lr=True, lr=lr)
]

# optimizer
optimizer = dict(type='SGD', lr=lr, weight_decay=1e-4, momentum=0.9)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'predictor': dict(fix_lr=True)}))

# runtime settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
