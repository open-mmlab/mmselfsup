_base_ = '../mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmcls.ToPIL', to_rgb=True),
    dict(type='mmcls.torchvision/Resize', size=224),
    dict(
        type='mmcls.torchvision/RandomCrop',
        size=224,
        padding=4,
        padding_mode='reflect'),
    dict(type='mmcls.torchvision/RandomHorizontalFlip', p=0.5),
    dict(type='mmcls.ToNumpy', to_rgb=True),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# model settings
model = dict(
    type='PixMIM',
    target_generator=dict(
        type='LowFreqTargetGenerator', radius=40, img_size=224),
)

# randomness
randomness = dict(seed=2, diff_rank_seed=True)
