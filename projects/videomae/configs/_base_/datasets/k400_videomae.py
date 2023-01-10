# dataset settings

dataset_type = 'mmaction.VideoDataset'
data_root = 'data/kinetics400/videos_train'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'

file_client_args = dict(
    io_backend='petrel',
    path_mapping=dict(
        {'data/kinetics400': 's3://openmmlab/datasets/action/Kinetics400'}))

# file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='mmaction.DecordInit', **file_client_args),
    dict(
        type='mmaction.SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1),
    dict(type='mmaction.DecordDecode'),
    dict(
        type='mmaction.MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='mmaction.Resize', scale=(224, 224), keep_ratio=False),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(
        type='VideoMAEMaskGenerator',
        input_size=(16, 224, 224),
        patch_size=16,
        tubelet_size=2,
        mask_ratio=0.9,
        mask_mode='tube'),
    dict(
        type='PackSelfSupInputs',
        key='imgs',
        algorithm_keys=['mask'],
        meta_keys=['img_shape', 'label'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
