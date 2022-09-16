# model settings
model = dict(
    type='MaskFeat',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(type='MaskFeatViT', arch='b', patch_size=16),
    hog_para=dict(nbins=9, pool=8, gaussian_window=16),
    head=dict(
        type='MaskFeatPretrainHead',
        predictor=dict(
            type='LinearNeck',
            in_channels=768,
            out_channels=108,
            with_avg_pool=False),
        loss=dict(type='MaskFeatReconstructionLoss')))
