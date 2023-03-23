import copy

# dataset settings
dataset_type = 'mmdet.CocoDataset'
# data_root = 'data/coco/'
data_root = '../data/coco/'
file_client_args = dict(backend='disk')
view_pipeline = [
    dict(
        type='RandomResizedCrop',
        size=224,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=1),
    dict(type='RandomSolarize', prob=0)
]
view_pipeline1 = copy.deepcopy(view_pipeline)
view_pipeline2 = copy.deepcopy(view_pipeline)
view_pipeline2[4]['prob'] = 0.1  # gaussian blur
view_pipeline2[5]['prob'] = 0.2  # solarization
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='MultiView',
        num_views=[1, 1],
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline))
