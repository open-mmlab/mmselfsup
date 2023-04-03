dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet'
extract_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmcls.ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackSelfSupInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=extract_pipeline),
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=extract_pipeline),
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# pooling cfg
pool_cfg = dict(type='AvgPool2d')
