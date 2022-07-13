# mmcls:: means we use the default settings from MMClassification
_base_ = [
    '../_base_/datasets/imagenet.py',
    'mmcls::_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'mmcls::_base_/default_runtime.py'
]

# model settings
custom_imports = dict(
    imports=['mmselfsup.models', 'mmselfsup.core'], allow_failed_imports=False)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        frozen_stages=12,
        avg_token=False,
        final_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint='')),
    neck=dict(type='mmselfsup.ClsBatchNormNeck', input_features=768),
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.01)]),
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ))

# dataset settings
file_client_args = dict(
    backend='memcached',
    server_list_cfg='/mnt/lustre/share/memcached_client/pcs_server_list.conf',
    client_cfg='/mnt/lustre/share_data/zhangwenwei/software/pymc/mc.conf',
    sys_path='/mnt/lustre/share_data/zhangwenwei/software/pymc')
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
train_dataloader = dict(batch_size=2048, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(type='mmselfsup.LARS', lr=6.4, weight_decay=0.0, momentum=0.9)
optim_wrapper = dict(
    type='AmpOptimWrapper', optimizer=optimizer, _delete_=True)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        by_epoch=True,
        begin=10,
        end=90,
        eta_min=0.0,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=90)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=10))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)
