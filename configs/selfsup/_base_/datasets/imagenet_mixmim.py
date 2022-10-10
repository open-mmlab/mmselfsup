# dataset settings
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/imagenet/': 'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/': 'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
    }))


dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
data_root = '/data/common/ImageNet/'
file_client_args = dict(backend='disk')

train_pipeline = [  # √
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="/home/nus-zwb/research/data/imagenet/train.txt",
        # ann_file="/mnt/lustre/zhaowangbo/research/2022ICLR/data/imagenet/meta/train.txt",
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
