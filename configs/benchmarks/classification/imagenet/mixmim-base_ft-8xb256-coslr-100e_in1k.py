_base_ = [
    '../_base_/datasets/imagenet-swin.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    'mmcls::_base_/default_runtime.py'
]


# load_from = "work_dirs/selfsup/mixmim_debug/epoch_300.pth"
# custom_imports = dict(imports=["/mnt/lustre/zhaowangbo/openmmlab/mmselfsup/mmselfsup"], allow_failed_imports=False)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='mmselfsup.MixMIMTransformerFinetune',
        arch='B',
        drop_rate=0.0,
        drop_path_rate=0.1),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8, num_classes=1000),
        dict(type='CutMix', alpha=1.0, num_classes=1000)
    ]))

# schedule settings
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=2e-3, model_type='mixmim', layer_decay_rate=0.7),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor')

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-8, # approach 0  # old setting 2e-4
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=2.5e-7 * 2048 / 512,    #  2.5e-7 * 2048 / 512
        by_epoch=True,
        begin=5,
        end=100,
        convert_to_iter_based=True)
]



file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/imagenet/': 'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/': 'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
    }))

dataset_type = 'ImageNet'
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
bgr_mean = preprocess_cfg['mean'][::-1]
bgr_std = preprocess_cfg['std'][::-1]

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
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=219,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    pin_memory=True,
    # collate_fn=dict(type="default_collate"),
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        # ann_file='meta/train.txt',
        ann_file="/mnt/lustre/zhaowangbo/research/2022ICLR/data/imagenet/meta/train.txt",
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    pin_memory=True,
    # collate_fn=dict(type="default_collate"),
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        # ann_file='meta/val.txt',
        ann_file="/mnt/lustre/zhaowangbo/research/2022ICLR/data/imagenet/meta/val.txt",
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)




default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=100))

randomness = dict(seed=0)


