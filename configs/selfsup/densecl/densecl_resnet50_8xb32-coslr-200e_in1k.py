_base_ = [
    '../_base_/models/densecl.py',
    '../_base_/datasets/imagenet_mocov2.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

find_unused_parameters = True

# runtime settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
