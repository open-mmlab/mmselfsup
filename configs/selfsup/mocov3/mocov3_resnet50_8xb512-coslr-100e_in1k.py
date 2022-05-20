_base_ = [
    '../_base_/models/mocov3_vit-small-p16.py',
    '../_base_/datasets/imagenet_mocov3.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset settings
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# the difference between ResNet50 and ViT pipeline is the `scale` in
# `RandomResizedCrop`, `scale=(0.08, 1.)` in ViT pipeline
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
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
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
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
    samples_per_gpu=512,
    workers_per_gpu=12,
    train=dict(
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch))

# MoCo v3 use the same momentum update method as BYOL
custom_hooks = [dict(type='MomentumUpdateHook')]

# the lr is set for batchs size 256
# ep100:    lr=0.6, wd=1e-6
# ep300:    lr=0.3, wd=1e-6
# ep1000:   lr=0.3, wd=1.5e-6
optimizer = dict(
    lr=9.6,
    momentum=0.9,
    weight_decay=1e-6,
    paramwise_options={
        '(bn|gn)(\\d+)?.(weight|bias)':
        dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True)
    })

checkpoint_config = dict(interval=10, max_keep_ckpts=3)

# runtime settings
runner = dict(max_epochs=100)

# fp16
fp16 = dict(loss_scale='dynamic')

# model settings
model = dict(
    base_momentum=0.9,  # 0.9 for 100ep and 300ep, 0.996 for 1000ep
    backbone=dict(
        type='ResNet',
        _delete_=True,
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        in_channels=2048,
        with_avg_pool=True,
        vit_backbone=False),
    head=dict(temperature=1.0))