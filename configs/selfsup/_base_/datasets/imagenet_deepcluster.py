# dataset settings
dataset_type = 'mmcls.ImageNet'
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
    dict(type='RandomGrayscale', p=0.2),
    dict(type='PackSelfSupInputs')
]

extract_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='PackSelfSupInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline))

# TODO: refactor the hook and modify the config
num_classes = 10000
custom_hooks = [
    dict(
        type='DeepClusterHook',
        extractor=dict(
            samples_per_gpu=128,
            workers_per_gpu=8,
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='meta/train.txt',
                data_prefix=dict(img='train/'),
                pipeline=extract_pipeline)),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=256),
        unif_sampling=True,
        reweight=False,
        reweight_pow=0.5,
        initial=True,  # call initially
        interval=1)
]
