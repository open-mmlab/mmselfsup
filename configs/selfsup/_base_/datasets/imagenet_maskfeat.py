# dataset settings
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/':
        'sproject:s3://openmmlab/datasets/classification/',
        'data/':
        'sproject:s3://openmmlab/datasets/classification/'
    }))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.5, 1.0),
        ratio=(0.75, 1.3333),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MaskfeatMaskGenerator', mask_window_size=14, mask_ratio=0.4),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
