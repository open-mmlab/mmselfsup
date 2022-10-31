_base_ = [
    '../_base_/models/vit-small-p16.py',
    'mmcls::_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    'mmcls::_base_/default_runtime.py',
]
# MoCoV3 ViT fine-tuning setting

# model settings
model = dict(
    _delete_=True,
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
        ]),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))

custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# optimizer
optimizer = dict(
    type='AdamW', lr=5e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=5.0),
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
        start_factor=1e-3,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=145,
        eta_min=1e-5,
        by_epoch=True,
        begin=5,
        end=150,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150)

# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))

randomness = dict(seed=0)
