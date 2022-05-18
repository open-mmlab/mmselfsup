# dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='RandomResizedCrop',
        size=192,
        scale=(0.67, 1.0),
        ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='BlockwiseMaskGenerator',
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(type='PackSelfSupInputs')
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline))
