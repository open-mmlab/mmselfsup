# model settings
model = dict(
    type='MAE',
    data_preprocessor=dict(
        mean=[124, 117, 104], std=[59, 58, 58], bgr_to_rgb=True),
    backbone=dict(type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        loss=dict(type='MAEReconstructionLoss')))
