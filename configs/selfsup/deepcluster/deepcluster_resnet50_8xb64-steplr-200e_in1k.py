_base_ = [
    '../_base_/models/deepcluster.py',
    '../_base_/datasets/imagenet_deepcluster.py',
    '../_base_/schedules/sgd_steplr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

model = dict(head=dict(num_classes={{_base_.num_classes}}))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-5,
    paramwise_options={'\\Ahead.': dict(momentum=0.)})

# learning policy
lr_config = dict(policy='step', step=[400])

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
