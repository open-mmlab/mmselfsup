# model settings
model = dict(
    type='MaskFeat',
    backbone=dict(
        type='MaskFeatViT',
        arch='b',
        patch_size=16,
        mask_ratio=0.4,
        drop_path_rate=0,
    ),
    head=dict(type='MaskFeatPretrainHead', hog_dim=108))
