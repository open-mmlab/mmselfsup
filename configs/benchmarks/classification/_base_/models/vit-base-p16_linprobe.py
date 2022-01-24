model = dict(
    type='MAEClassification',
    backbone=dict(
        type='MAEClsViT',
        arch='b',
        patch_size=16,
        global_pool=True,
        finetune=False,
        final_norm=False),
    head=dict(type='MAELinprobeHead', num_classes=1000, embed_dim=768),
    finetune=False)
