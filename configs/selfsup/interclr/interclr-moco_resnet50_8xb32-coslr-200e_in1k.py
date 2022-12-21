_base_ = [
    '../_base_/models/interclr-moco.py',
    '../_base_/datasets/imagenet_interclr-moco.py',
    '../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    memory_bank=dict(num_classes={{_base_.num_classes}}),
    num_classes={{_base_.num_classes}},
)

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
