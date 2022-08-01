_base_ = [
    '../_base_/models/mocov3_vit-small-p16.py',
    '../_base_/datasets/imagenet_mocov3.py',
    '../_base_/schedules/adamw_coslr-300e_in1k.py',
    '../_base_/default_runtime.py',
]

# dataset settings
# the difference between ResNet50 and ViT pipeline is the `scale` in
# `RandomResizedCrop`, `scale=(0.08, 1.)` in ViT pipeline
view_pipeline1 = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.08, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=1.),
    dict(type='RandomSolarize', prob=0.),
    dict(type='RandomFlip', prob=0.5),
]
view_pipeline2 = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.08, 1.),
        backend='pillow'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.1),
    dict(type='RandomSolarize', prob=0.2),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(
        type='MultiView',
        num_views=[1, 1],
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# optimizer
optimizer = dict(type='AdamW', lr=2.4e-3, weight_decay=0.1)
optim_wrapper = dict(
    type='AmpOptimWrapper', loss_scale='dynamic', optimizer=optimizer)
find_unused_parameters = True

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]

# runtime settings
# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
