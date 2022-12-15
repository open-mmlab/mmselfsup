_base_ = '../mae/mae_vit-base-p16_8xb512-amp-coslr-400e_in1k.py'

# model settings
model = dict(
    type='EVA',
    backbone=dict(type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75),
    neck=dict(type='MAEPretrainDecoder', predict_feature_dim=512),
    head=dict(
        _delete_=True,
        type='EVAPretrainHead',
        loss=dict(
            type='CosineSimilarityLoss', shift_factor=1.0, scale_factor=1.0
        ),  # to keep the same with the official implement of EVA
    ),
    target_generator=dict(
        type='CLIPGenerator',
        tokenizer_path=  # noqa
        'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/clip_vit_base_16.pth.tar'  # noqa
    ),
    init_cfg=None)

# dataset 16 x 256
# NUS dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = '/data/common/ImageNet/'
file_client_args = dict(backend='disk')
ann_file = '/home/nus-zwb/research/data/imagenet/meta/train.txt'

train_pipeline = [
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
    batch_size=256,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

find_unused_parameters = True
