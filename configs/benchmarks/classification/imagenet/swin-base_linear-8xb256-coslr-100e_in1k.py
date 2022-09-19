# mmcls:: means we use the default settings from MMClassification
_base_ = [
    'mmcls::_base_/models/swin_transformer/base_224.py',
    '../_base_/datasets/imagenet-swin.py',
    'mmcls::_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'mmcls::_base_/default_runtime.py'
]

# model settings
custom_imports = dict(
    imports=['mmselfsup.models', 'mmselfsup.engine'],
    allow_failed_imports=False)
# model settings
model = dict(
    backbone=dict(
        img_size=192,
        drop_path_rate=0.1,
        stage_cfgs=dict(block_cfgs=dict(window_size=6)),
        frozen_stages=3,
        out_indices=[3]))

# dataset settings
file_client_args = dict(backend='disk')
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
