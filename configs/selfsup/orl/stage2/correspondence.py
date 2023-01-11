_base_ = [
    '../../_base_/default_runtime.py',
]
dist_params = dict(backend='nccl', port=29500)
# model settings
model = dict(
    type='Correspondence',
    base_momentum=0.99,
    pretrained=None,
    knn_image_num=10,
    topk_bbox_ratio=0.1,
    # data_preprocessor=dict(
    #     mean=(123.675, 116.28, 103.53),
    #     std=(58.395, 57.12, 57.375),
    #     bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        with_bias=False,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=False,
            with_avg_pool=False),
        loss=dict(type='CosineSimilarityLoss')),
)
# dataset settings
train_knn_json = '../data/coco/meta/train2017_10nn_instance.json'
train_ss_json = '../data/coco/meta/train2017_selective_search_proposal.json'
data_train_root = '../data/coco/train2017'
dataset_type = 'CorrespondDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset_dict = dict(
    type=dataset_type,
    knn_json_file=train_knn_json,
    ss_json_file=train_ss_json,
    root=data_train_root,
    part=0,  # [0, num_parts)
    num_parts=1,  # process the whole dataset
    data_len=118287,
    # **data_source_cfg),
    norm_cfg=img_norm_cfg,
    patch_size=224,
    min_size=96,
    max_ratio=3,
    max_iou_thr=0.5,
    topN=100,
    knn_image_num=10,
    topk_bbox_ratio=0.1)
val_dataloader = dict(
    # support single-image single-gpu inference only
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dataset_dict)

# additional hooks
update_interval = 1  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]
# Amp optimizer
optimizer = dict(type='SGD', lr=0.4, weight_decay=0.0001, momentum=0.9)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=optimizer,
    accumulative_counts=update_interval,
)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=4,
    warmup_ratio=0.0001,  # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 800
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs)
