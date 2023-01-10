_base_ = ['./mmaction2_default_runtime.py']

# _base_ = ['mmaction::_base_/default_runtime.py']

# custom_imports = dict(
#     imports=['mmaction.datasets.transforms'],
#     allow_failed_imports=False)

custom_imports = dict(imports=['mmaction'], allow_failed_imports=False)

# model settings
model = dict(
    type='mmaction.Recognizer3D',
    backbone=dict(
        type='mmaction.VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='mmaction.TimeSformerHead',
        num_classes=400,
        in_channels=768,
        average_clips='prob',
        loss_cls=dict(type='mmaction.CrossEntropyLoss')),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'mmaction.VideoDataset'
data_root_val = 'data/kinetics400/videos_val'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'

file_client_args = dict(
    io_backend='petrel',
    path_mapping=dict(
        {'data/kinetics400': 's3://openmmlab/datasets/action/Kinetics400'}))

test_pipeline = [
    dict(type='mmaction.DecordInit', **file_client_args),
    dict(
        type='mmaction.SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True),
    dict(type='mmaction.DecordDecode'),
    dict(type='mmaction.Resize', scale=(-1, 224)),
    dict(type='mmaction.ThreeCrop', crop_size=224),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(type='mmaction.PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='mmaction.DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = dict(type='mmaction.AccMetric')
test_cfg = dict(type='mmaction.TestLoop')
