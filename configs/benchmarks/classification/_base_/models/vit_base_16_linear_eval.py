model = dict(
    type='MAELinearClassification',
    backbone=dict(
        type='MAEFinetuneViT',
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        global_pool=True,
        finetune=False),
    head=dict(type='MAELinearEvalHead', num_classes=1000, embed_dim=768))
