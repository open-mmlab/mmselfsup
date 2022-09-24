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
    hog_para=dict(
        nbins=9,  # Number of bin. Defaults to 9.
        pool=8,  # Number of cell. Defaults to 8.
        gaussian_window=16  # Size of gaussian kernel. Defaults to 16.
    ))
