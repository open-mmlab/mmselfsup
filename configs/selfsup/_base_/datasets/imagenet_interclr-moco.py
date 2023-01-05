# dataset settings
data_source = 'ImageNet'
train_dataset_type = 'MultiViewDataset'
extract_dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.5),
    dict(type='RandomHorizontalFlip'),
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
    samples_per_gpu=32,  # total 32*8=256
    replace=True,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=train_dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch))

# additional hooks
num_classes = 10000
custom_hooks = [
    dict(
        type='InterCLRHook',
        extractor=dict(
            samples_per_gpu=256,
            workers_per_gpu=8,
            dataset=dict(
                type=extract_dataset_type,
                data_source=dict(
                    type=data_source,
                    data_prefix='data/imagenet/train',
                    ann_file='data/imagenet/meta/train.txt',
                ),
                pipeline=extract_pipeline,
                prefetch=prefetch),
            prefetch=prefetch,
            img_norm_cfg=img_norm_cfg),
        clustering=dict(type='Kmeans', k=num_classes, pca_dim=-1),  # no pca
        centroids_update_interval=10,  # iter
        deal_with_small_clusters_interval=1,
        evaluate_interval=50,
        warmup_epochs=0,
        init_memory=True,
        initial=True,  # call initially
        online_labels=True,
        interval=10)  # same as the checkpoint interval
]
