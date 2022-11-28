_base_ = [
    '../_base_/models/mixmim.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]



# dataset settings
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         '.data/imagenet/': 'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
#         'data/imagenet/': 'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
#     }))

# data_root = 'data/imagenet/'

dataset_type = 'mmcls.ImageNet'

file_client_args = dict(backend='disk')
data_root = '/data/personal/nus-zwb/ImageNet/'
train_ann_file = '/home/nus-zwb/research/data/imagenet/meta/train.txt'
val_ann_file = '/home/nus-zwb/research/data/imagenet/meta/val.txt'

# data_root = '/data/common/ImageNet/'
# file_client_args = dict(backend='disk')

train_pipeline = [  # √
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file="/home/nus-zwb/research/data/imagenet/train.txt",
        ann_file=train_ann_file,
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))


#train_dataloader = dict(batch_size=2, num_workers=1)

# optimizer wrapper
optimizer = dict(
    type='AdamW', lr= 1.5e-4 * (8 * 128 / 256), betas=(0.9, 0.95), weight_decay=0.05)  # 4096 = 8GPU * 512batchsize
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'ln': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0)
        }))


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]


train_cfg = dict(max_epochs=300)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1))

# randomness
randomness = dict(seed=0, diff_rank_seed=True)  # √



vis_backends = [dict(type='TensorboardVisBackend'), dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)