_base_ = 'mmdet::mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# https://github.com/open-mmlab/mmdetection/blob/dev-3.x/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(frozen_stages=-1, norm_cfg=norm_cfg, norm_eval=False),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
