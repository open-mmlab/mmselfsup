# model settings
model = dict(
    type='MaskFeat',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(type='MaskFeatViT', arch='b', patch_size=16),
    neck=dict(
        type='LinearNeck',
        in_channels=768,
        out_channels=108,
        with_avg_pool=False,
        init_cfg=dict(type='TruncNormal', layer='Linear', std=0.02, bias=0)),
    head=dict(
        type='MaskFeatPretrainHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    target_generator=dict(
        type='HOGGenerator', nbins=9, pool=8, gaussian_window=16))
