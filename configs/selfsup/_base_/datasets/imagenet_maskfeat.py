# dataset settings
<<<<<<< HEAD
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
=======
data_source = 'ImageNet'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
>>>>>>> upstream/master
        size=224,
        scale=(0.5, 1.0),
        ratio=(0.75, 1.3333),
        interpolation='bicubic'),
<<<<<<< HEAD
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='BEiTMaskGenerator',
        input_size=14,
        num_masking_patches=78,
        min_num_patches=15,
    ),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

# for visualization
vis_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(224, 224), backend='pillow'),
    dict(
        type='BEiTMaskGenerator',
        input_size=14,
        num_masking_patches=78,
        min_num_patches=15,
    ),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]
=======
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

train_pipeline.append(dict(type='MaskFeatMaskGenerator', mask_ratio=0.4))

# dataset summary
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt'),
        pipeline=train_pipeline,
        prefetch=prefetch))
>>>>>>> upstream/master
