# mmcls:: means we use the default settings from MMClassification
_base_ = [
    'mmcls::_base_/datasets/imagenet_bs64_swin_224.py',
    'mmcls::_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'mmcls::_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        avg_token=True,
        final_norm=False,
        init_cfg=dict(type='Pretrained', checkpoint='')),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=2e-5),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8, num_classes=1000),
        dict(type='CutMix', alpha=1.0, num_classes=1000)
    ]))

# dataset settings
# file_client_args = dict(
#     backend='memcached',
#     server_list_cfg='/mnt/lustre/share/memcached_client/pcs_server_list.conf',
#     client_cfg='/mnt/lustre/share_data/zhangwenwei/software/pymc/mc.conf',
#     sys_path='/mnt/lustre/share_data/zhangwenwei/software/pymc')
file_client_args = dict(
    backend='petrel',
    # 因为petreloss.conf设置sproject为默认，此处可省略前缀
    path_mapping=dict({
        './data/imagenet':
        's3://openmmlab/datasets/classification/imagenet',
        'data/imagenet':
        's3://openmmlab/datasets/classification/imagenet'
    }))

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
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(batch_size=128, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=128, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer wrapper
custom_imports = dict(imports='mmselfsup.engine', allow_failed_imports=False)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=2e-3,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
        model_type='vit',  # layer-wise lr decay type
        layer_decay_rate=0.65),  # layer-wise lr decay factor
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# runtime settings
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3))

train_cfg = dict(by_epoch=True, max_epochs=100)

randomness = dict(seed=0)
