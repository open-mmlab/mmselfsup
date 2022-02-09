data_source = 'ImageNet'
dataset_type = 'SingleViewDataset'
name = 'imagenet_val'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    extract=dict(
        type='SingleViewDataset',
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/val',
            ann_file='data/imagenet/meta/val.txt',
        ),
        pipeline=[
            dict(type='Resize', size=256),
            dict(type='CenterCrop', size=224),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))
