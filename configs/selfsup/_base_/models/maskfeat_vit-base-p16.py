# model settings
model = dict(
    type='MaskFeat',
    backbone=dict(
        type='MaskFeatViT',
        arch='b',
        patch_size=16,
        drop_path_rate=0,
    ),
    head=dict(type='MaskFeatPretrainHead', hog_dim=108),
    hog_para=dict(nbins=9, pool=8, gaussian_window=16))
