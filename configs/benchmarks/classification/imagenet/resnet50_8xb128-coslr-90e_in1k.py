_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet.py',
    '../_base_/schedules/sgd_coslr-100e.py',
    '../_base_/default_runtime.py',
]

model = dict(backbone=dict(frozen_stages=4))

evaluation = dict(interval=1, topk=(1, 5))

# mocov3-ResNet50 setting
# base lr is 0.1 for batch size 256
optimizer = dict(type='SGD', lr=0.4, momentum=0.9, weight_decay=0.)
data = dict(samples_per_gpu=128, workers_per_gpu=8)  # total 128*8=1024, 8GPU linear cls
# runtime settings
runner = dict(max_epochs=90)


# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10)
