_base_ = [
    '../_base_/models/deepcluster.py',
    '../_base_/datasets/imagenet_deepcluster.py',
    '../_base_/schedules/sgd_steplr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

model = dict(head=dict(num_classes={{_base_.num_classes}}))

# optimizer
optimizer = dict(type='SGD', lr=0.1, weight_decay=1e-5, momentum=0.9)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(momentum=0.)}))

# learning rate scheduler
param_scheduler = [
    dict(type='MultiStepLR', by_epoch=True, milestones=[400], gamma=0.1)
]

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
