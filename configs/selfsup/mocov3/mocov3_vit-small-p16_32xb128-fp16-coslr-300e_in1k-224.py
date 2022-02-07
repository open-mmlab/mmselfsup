_base_ = [
    '../_base_/models/mocov3_vit-small-p16.py',
    '../_base_/datasets/imagenet_mocov3.py',
    '../_base_/schedules/adamw_coslr-300e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset settings
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# the difference between ResNet50 and ViT pipeline is the `scale` in
# `RandomResizedCrop`, `scale=(0.08, 1.)` in ViT pipeline
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.08, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=1.),
    dict(type='Solarization', p=0.),
    dict(type='RandomHorizontalFlip'),
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.08, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.1),
    dict(type='Solarization', p=0.2),
    dict(type='RandomHorizontalFlip'),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline1.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])
    train_pipeline2.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=128,
    train=dict(pipelines=[train_pipeline1, train_pipeline2]))

# MoCo v3 use the same momentum update method as BYOL
custom_hooks = [dict(type='MomentumUpdateHook')]

# optimizer
optimizer = dict(type='AdamW', lr=2.4e-3, weight_decay=0.1)

# fp16
fp16 = dict(loss_scale='dynamic')

# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)
