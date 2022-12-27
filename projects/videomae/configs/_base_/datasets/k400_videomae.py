# dataset settings

dataset_type = 'mmaction2.VideoDataset'
data_root = 'data/kinetics400/videos_train'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='mmaction2.DecordInit', **file_client_args),
    dict(
        type='mmaction2.SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1),
    dict(type='mmaction2.DecordDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    dict(
        type='mmaction2.MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    # dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='mmaction2.FormatShape', input_format='NCTHW'),
    dict(
        type='VideoMAEMaskGenerator',
        input_size=(8, 224, 224),
        mask_ratio=0.9,
        mask_mode='tube'),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['video_path'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
