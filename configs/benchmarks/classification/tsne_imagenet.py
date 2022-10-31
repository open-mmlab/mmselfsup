dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')
name = 'imagenet_val'

extract_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmcls.ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackSelfSupInputs'),
]

extract_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=extract_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# pooling cfg
pool_cfg = dict(type='MultiPooling', in_indices=(1, 2, 3, 4))
