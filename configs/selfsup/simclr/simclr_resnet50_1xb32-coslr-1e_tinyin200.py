_base_ = 'simclr_resnet50_8xb32-coslr-200e_in1k.py'

# dataset

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='mmcls.ImageNet',
        data_root='../data/tiny-imagenet-200/',
        ann_file='train.txt',
        data_prefix=dict(img_path='train/'),
        ))
# optimizer
optimizer = dict(
    lr=0.3 * ((32 * 1) / (32 * 8)),
)

runner = dict(max_epochs=20)