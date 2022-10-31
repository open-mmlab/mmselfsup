dataset_type = 'ImageList'
data_root = 'data/VOCdevkit/VOC2007/'
file_client_args = dict(backend='disk')

split_at = [5011]
split_name = ['voc07_trainval', 'voc07_test']

extract_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='mmcls.ResizeEdge', scale=256),
    dict(type='Resize', scale=224),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

extract_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    extract_dataset=dict(
        type=dataset_type,
        ann_file='Lists/trainvaltest.txt',
        data_root=data_root,
        data_prefix='JPEGImages/',
        pipeline=extract_pipeline))

# pooling cfg
pool_cfg = dict(
    type='MultiPooling', pool_type='specified', in_indices=(0, 1, 2, 3, 4))
