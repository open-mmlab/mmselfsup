# dataset settings
dataset_type = 'Places205'
data_root = 'data/Places205/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=256),
    dict(type='CenterCrop', crop_size=256),
    dict(type='RandomCrop', size=224),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=256),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackSelfSupInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainvalsplit_places205/train_places205.csv',
        data_prefix=dict(
            img_path='data/vision/torralba/deeplearning/images256/'),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainvalsplit_places205/val_places205.csv',
        data_prefix=dict(
            img_path='data/vision/torralba/deeplearning/images256/'),
        pipeline=test_pipeline))
val_evaluator = dict(type='Accuracy', top_k=(1, 5))
