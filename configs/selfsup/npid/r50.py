_base_ = '../../base.py'
# model settings
model = dict(
    type='NPID',
    neg_num=65536,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='LinearNeck',
        in_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.07),
    memory_bank=dict(
        type='SimpleMemory', length=1281167, feat_dim=128, momentum=0.5))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/imagenet/meta/train.txt'
data_train_root = 'data/imagenet/train'
dataset_type = 'NPIDDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=32,  # total 32*8
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9, nesterov=False)
# learning policy
lr_config = dict(policy='step', step=[120, 160])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
