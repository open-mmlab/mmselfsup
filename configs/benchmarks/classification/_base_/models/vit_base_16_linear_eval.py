model = dict(
    type='MAELinearClassification',
    backbone=dict(
        type='MAEClsViT',
        arch='b',
        patch_size=16,
        global_pool=True,
        finetune=False),
    head=dict(type='MAELinearEvalHead', num_classes=1000, embed_dim=768))
