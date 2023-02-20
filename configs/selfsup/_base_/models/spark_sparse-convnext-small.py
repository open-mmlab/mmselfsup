# model settings
model = dict(
    type='SparK',
    input_size=224,
    downsample_raito=32,
    mask_ratio=0.6,
    mask_ratio2=0.6,
    uniform=False,
    enc_dec_norm_cfg=dict(type='SparseLN2d', eps=1e-6),
    enc_dec_norm_dim=768,
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True),
    backbone=dict(
        type='SparseConvNeXt',
        arch='small',
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        gap_before_output=False,
        use_grn=True),
    neck=dict(
        type='SparKLightDecoder',
        feature_dim=512,
        upsample_ratio=32,  # equal to downsample_raito
        mid_channels=0,
        last_act=False),
    head=dict(
        type='SparKPretrainHead',
        loss=dict(type='PixelReconstructionLoss', criterion='L2')))
