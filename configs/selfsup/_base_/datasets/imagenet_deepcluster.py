# dataset settings
data_source = 'ImageNet'
dataset_type = 'DeepClusterDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=1.0,
        hue=0.5),
    dict(type='RandomGrayscale', p=0.2),
]
extract_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])
    extract_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=64,  # 64*8
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch))

# additional hooks
num_classes = 10000
custom_hooks = [
    dict(
        type='DeepClusterHook',
        extractor=dict(
            samples_per_gpu=128,
            workers_per_gpu=8,
            dataset=dict(
                type=dataset_type,
                data_source=dict(
                    type=data_source,
                    data_prefix='data/imagenet/train',
                    ann_file='data/imagenet/meta/train.txt',
                ),
                pipeline=extract_pipeline,
                prefetch=prefetch),
            prefetch=prefetch,
            img_norm_cfg=img_norm_cfg),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=256),
        unif_sampling=True,
        reweight=False,
        reweight_pow=0.5,
        initial=True,  # call initially
        interval=1)
]
