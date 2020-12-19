_base_ = '../../base.py'
# model settings
model = dict(
    type='SimSiam',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckSimCLR',  # SimCLR non-linear neck
        in_channels=512,
        hid_channels=2048,
        out_channels=2048,
        num_layers=3,
        with_avg_pool=True),
    head=dict(
        type='SimSiamPredictHead',
        predictor=dict(
            type='NonLinearNeckV2',
            in_channels=2048,
            hid_channels=512,
            out_channels=2048,
            with_avg_pool=False),
        loss=dict(type='NegativeCosineSimilarityLoss')
    )
)
# dataset settings
data_source_cfg = dict(type='Cifar10', root='data/cifar/', has_labels=False)
dataset_type = 'ContrastiveDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_size = (32, 32)
train_pipeline = [
    dict(type='RandomResizedCrop', size=img_size, scale=(0.2, 1.0)),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    # We do not use blur augmentation for cifar experiments
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='GaussianBlur',
    #             sigma_min=0.1,
    #             sigma_max=2.0)
    #     ],
    #     p=0.5),
]
test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=512,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline)
)
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=5e-4, momentum=0.9)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)
checkpoint_config = dict(interval=100)
# runtime settings
total_epochs = 800
