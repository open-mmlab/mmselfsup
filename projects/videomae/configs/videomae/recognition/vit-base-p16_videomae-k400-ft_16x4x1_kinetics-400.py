_base_ = ['./vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py']

custom_imports = dict(
    imports=['mmaction'],
    # imports=['mmaction.datasets.transforms'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='mmaction.Recognizer3D',
    backbone=dict(
        type='mmaction.VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='mmaction.TimeSformerHead',
        num_classes=400,
        in_channels=384,
        average_clips='prob',
        loss_cls=dict(type='mmaction.CrossEntropyLoss')),
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'mmaction.VideoDataset'
data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'

file_client_args = dict(
    io_backend='petrel',
    path_mapping=dict(
        {'data/kinetics400': 's3://openmmlab/datasets/action/Kinetics400'}))

train_pipeline = [
    dict(type='mmaction.DecordInit', **file_client_args),
    dict(
        type='mmaction.SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=1),
    dict(type='mmaction.DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='mmaction.DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

val_pipeline = [
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

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='mmaction.DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='mmaction.DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

val_evaluator = dict(type='mmaction.AccMetric')
test_evaluator = dict(type='mmaction.AccMetric')

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=3)
val_cfg = dict(type='mmaction.ValLoop')
test_cfg = dict(type='mmaction.TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4, nesterov=True),
    paramwise_cfg=dict(
        custom_keys={
            '.backbone.cls_token': dict(decay_mult=0.0),
            '.backbone.pos_embed': dict(decay_mult=0.0),
            '.backbone.time_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=15,
        by_epoch=True,
        milestones=[5, 10],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=5), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
