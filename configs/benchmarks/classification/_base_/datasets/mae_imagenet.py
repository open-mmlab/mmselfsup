# dataset settings
data_source = 'ImageNet'
dataset_type = 'MAEFtDataset'
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
    }))
train_pipeline = [
    dict(
        type='MAEFtAugment',
        input_size=224,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_train=True),
]
test_pipeline = [
    dict(
        type='MAEFtAugment',
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
]

prefetch = False
# dataset summary
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
            file_client_args=file_client_args,
        ),
        pipeline=train_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/val',
            ann_file='data/imagenet/meta/val.txt',
            file_client_args=file_client_args,
        ),
        pipeline=test_pipeline,
        prefetch=prefetch),
)

evaluation = dict(interval=10, topk=(1, 5))
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
