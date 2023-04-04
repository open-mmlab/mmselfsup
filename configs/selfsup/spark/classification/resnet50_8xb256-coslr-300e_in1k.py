_base_ = [
    'mmcls::_base_/models/resnet50.py',
    'mmcls::_base_/datasets/imagenet_bs256_rsb_a12.py',
    'mmcls::_base_/default_runtime.py'
]
# modification is based on ResNets RSB settings

# dataset settings
data_root = '/nvme/dataset/classification/imagenet/'
train_dataloader = dict(dataset=dict(data_root=data_root,))
val_dataloader = dict(dataset=dict(data_root=data_root))

# model settings
model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        drop_path_rate=0.05,
    ),
    head=dict(
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, use_sigmoid=True)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ]))

custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

# schedule settings
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='Lamb',
        lr=0.016,
        weight_decay=0.02,
        model_type='resnet',
        layer_decay_rate=0.7),
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor',
    paramwise_cfg=dict(
        custom_keys={
            'bias': dict(decay_mult=0.),
            'bn': dict(decay_mult=0.),
            'downsample.1': dict(decay_mult=0.),
        }))

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=295,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=300)
]
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict()
test_cfg = dict()

default_hooks = dict(
    # only keeps the latest 2 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)
