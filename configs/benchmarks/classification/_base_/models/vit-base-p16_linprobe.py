model = dict(
    type='Classification',
    backbone=dict(type='MAEClsViT', arch='b', patch_size=16, final_norm=False),
    head=dict(type='MAELinprobeHead', num_classes=1000, embed_dim=768))
