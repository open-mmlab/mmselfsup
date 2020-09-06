_base_ = '../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[0, 1, 2, 3, 4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='MultiClsHead',
        pool_type='specified',
        in_indices=[0, 1, 2, 3, 4],
        with_last_layer_unpool=False,
        backbone='resnet50',
        norm_cfg=dict(type='SyncBN', momentum=0.1, affine=False),
        num_classes=205))
# dataset settings
data_source_cfg = dict(
    type='Places205',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/places205/meta/train_labeled.txt'
data_train_root = 'data/places205/train'
data_test_list = 'data/places205/meta/val_labeled.txt'
data_test_root = 'data/places205/val'
dataset_type = 'ClassificationDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=256),
    dict(type='RandomCrop', size=224),
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
    imgs_per_gpu=32,  # total 32x8=256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))
# additional hooks
custom_hooks = [
    dict(
        type='ValidateHook',
        dataset=data['val'],
        initial=True,
        interval=10,
        imgs_per_gpu=32,
        workers_per_gpu=4,
        eval_param=dict(topk=(1, )))
]
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(norm_decay_mult=0.),
    nesterov=True)
# learning policy
lr_config = dict(policy='step', step=[7, 14, 21])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 28
