_base_ = [
    '../_base_/models/odc.py',
    '../_base_/datasets/imagenet_odc.py',
    '../_base_/schedules/sgd_steplr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    head=dict(num_classes={{_base_.num_classes}}),
    memory_bank=dict(num_classes={{_base_.num_classes}}),
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.06,
    momentum=0.9,
    weight_decay=1e-5,
    paramwise_options={'\\Ahead.': dict(momentum=0.)})

# learning policy
lr_config = dict(policy='step', step=[400], gamma=0.4)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=440)
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
