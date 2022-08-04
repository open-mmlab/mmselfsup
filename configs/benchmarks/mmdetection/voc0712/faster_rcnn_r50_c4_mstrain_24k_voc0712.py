_base_ = [
    '../_base_/models/faster_rcnn_r50_c4.py',
    '../_base_/schedules/schedule_24k.py',
    '../_base_/default_runtime.py',
    '../_base_/datasets/voc0712.py',
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(frozen_stages=-1, norm_cfg=norm_cfg, norm_eval=False),
    roi_head=dict(
        shared_head=dict(
            type='ResLayerExtraNorm', norm_cfg=norm_cfg, norm_eval=False),
        bbox_head=dict(num_classes=20)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

train_dataloader = dict(
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='VOCDataset',
                data_root=data_root,
                ann_file='VOC2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2007/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline),
            dict(
                type='VOCDataset',
                data_root=data_root,
                ann_file='VOC2012/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2012/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline)
        ]))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

checkpoint_config = dict(by_epoch=False, interval=2000)

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

custom_imports = dict(
    imports=['mmselfsup.evaluation.functional.res_layer_extra_norm'],
    allow_failed_imports=False)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

default_scope = 'mmdet'
