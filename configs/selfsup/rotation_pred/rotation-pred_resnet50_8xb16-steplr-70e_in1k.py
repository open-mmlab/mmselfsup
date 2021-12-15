_base_ = [
    '../_base_/models/rotation-pred.py',
    '../_base_/datasets/imagenet_rotation-pred.py',
    '../_base_/schedules/sgd_steplr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=1e-4)

# learning policy
lr_config = dict(
    policy='step',
    step=[30, 50],
    warmup='linear',
    warmup_iters=5,  # 5 ep
    warmup_ratio=0.1,
    warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=70)
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
