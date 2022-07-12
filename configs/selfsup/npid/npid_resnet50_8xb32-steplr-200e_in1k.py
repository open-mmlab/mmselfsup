_base_ = [
    '../_base_/models/npid.py',
    '../_base_/datasets/imagenet_npid.py',
    '../_base_/schedules/sgd_steplr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# runtime settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
