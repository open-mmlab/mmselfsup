# dataset settings
dataset_type = 'DeepClusterImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotation', degrees=2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=1.0,
        hue=0.5),
    dict(type='RandomGrayscale', prob=0.2, keep_channels=True),
    dict(
        type='PackSelfSupInputs',
        pseudo_label_keys=['clustering_label'],
        meta_keys=['img_path'])
]

extract_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DeepClusterSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

num_classes = 10000
custom_hooks = [
    dict(
        type='DeepClusterHook',
        extract_dataloader=dict(
            batch_size=128,
            num_workers=8,
            persistent_workers=True,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='meta/train.txt',
                data_prefix=dict(img_path='train/'),
                pipeline=extract_pipeline)),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=256),
        unif_sampling=True,
        reweight=False,
        reweight_pow=0.5,
        initial=True,  # call initially
        interval=1)
]
