# dataset settings
data_source = 'ImageNet'
dataset_type = 'MultiViewDataset'
num_crops = [2, 6]
color_distort_strength = 1.0
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.14, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8 * color_distort_strength,
                contrast=0.8 * color_distort_strength,
                saturation=0.8 * color_distort_strength,
                hue=0.2 * color_distort_strength)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type='RandomHorizontalFlip', p=0.5),
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=96, scale=(0.05, 0.14)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8 * color_distort_strength,
                contrast=0.8 * color_distort_strength,
                saturation=0.8 * color_distort_strength,
                hue=0.2 * color_distort_strength)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type='RandomHorizontalFlip', p=0.5),
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
    samples_per_gpu=32,  # total 32*8=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
        ),
        num_views=num_crops,
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch))
