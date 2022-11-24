# mmcls:: means we use the default settings from MMClassification
_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'mmcls::_base_/default_runtime.py'
]

data_preprocessor = dict(
    num_classes=1000,
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True,
)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='BEiT',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_rate=0.,
        drop_path_rate=0.,
        final_norm=True,
        avg_token=False,
        frozen_stages=12,
        output_cls_token=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False),
    neck=dict(type='TransformerTokenMergeNeck', mode='concat'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.01)]),
)

# dataset settings
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/':
        'sproject:s3://openmmlab/datasets/classification/',
        'data/':
        'sproject:s3://openmmlab/datasets/classification/'
    }))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]
train_dataloader = dict(batch_size=128, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=128, dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(type='AdamW', lr=4e-3, weight_decay=1e-4)
optim_wrapper = dict(
    _delete_=True, type='AmpOptimWrapper', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=100,
        eta_min=0.0,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=100)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
