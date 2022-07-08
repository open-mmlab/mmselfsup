_base_ = [
    '../_base_/models/swav.py',
    '../_base_/datasets/imagenet_swav_mcrop-2-6.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(head=dict(loss=dict(num_crops={{_base_.num_crops}})))

# additional hooks
custom_hooks = [
    dict(
        type='SwAVHook',
        priority='VERY_HIGH',
        batch_size={{_base_.train_dataloader.batch_size}},
        epoch_queue_starts=15,
        crops_for_assign=[0, 1],
        feat_dim=128,
        queue_length=3840,
        frozen_layers_cfg=dict(prototypes=5005))
]

# dataset summary
data = dict(num_views={{_base_.num_crops}})

# optimizer
optimizer = dict(type='LARS', lr=0.6)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=200,
        eta_min=6e-4,
        by_epoch=True,
        begin=0,
        end=200,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

find_unused_parameters = True
