_base_ = [
    '../_base_/models/mocov2.py',
    '../_base_/datasets/imagenet_mocov2.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
