# mmcls:: means we use the default settings from MMClassification
_base_ = 'vit-base-p16_ft-8xb128-coslr-100e_in1k.py'
# maskfeat fine-tuning setting

# model settings
model = dict(
    head=dict(init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=2e-5),
    ]))

# dataset
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(batch_size=256, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=256, dataset=dict(pipeline=test_pipeline))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader

# optimizer wrapper
optim_wrapper = dict(optimizer=dict(lr=0.008))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=80,
        by_epoch=True,
        begin=20,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(max_epochs=100)
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

randomness = dict(seed=0)
