# dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

num_crops = [2, 6]
color_distort_strength = 1.0
view_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.14, 1.)),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8 * color_distort_strength,
                contrast=0.8 * color_distort_strength,
                saturation=0.8 * color_distort_strength,
                hue=0.2 * color_distort_strength)
        ],
        prob=0.8),
    dict(type='RandomGrayscale', prob=0.2),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]
view_pipeline2 = [
    dict(type='RandomResizedCrop', size=96, scale=(0.05, 0.14)),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8 * color_distort_strength,
                contrast=0.8 * color_distort_strength,
                saturation=0.8 * color_distort_strength,
                hue=0.2 * color_distort_strength)
        ],
        prob=0.8),
    dict(type='RandomGrayscale', prob=0.2),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiView',
        num_views=num_crops,
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackSelfSupInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline))
