# dataset settings
data_source = 'ImageNet'
dataset_type = 'RelativeLocDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='Resize', size=292),
    dict(type='RandomCrop', size=255),
    dict(type='RandomGrayscale', p=0.66),
]
test_pipeline = [
    dict(type='Resize', size=292),
    dict(type='CenterCrop', size=255),
]
format_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]

# prefetch
prefetch = False

# dataset summary
data = dict(
    samples_per_gpu=64,  # 64 x 8 = 512
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
        ),
        pipeline=train_pipeline,
        format_pipeline=format_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/val',
            ann_file='data/imagenet/meta/val.txt',
        ),
        pipeline=test_pipeline,
        format_pipeline=format_pipeline,
        prefetch=prefetch))
