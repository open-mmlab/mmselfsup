# dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=292),
    dict(type='RandomCrop', size=255),
    dict(type='RandomGrayscale', prob=0.66, keep_channels=True),
    dict(type='RandomPatchWithLabels'),
    dict(
        type='PackSelfSupInputs',
        pseudo_label_keys=['patch_box', 'patch_label', 'unpatched_img'],
        meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
