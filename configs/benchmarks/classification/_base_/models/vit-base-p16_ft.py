model = dict(
    type='Classification',
    backbone=dict(
        type='MIMVisionTransformer',
        arch='b',
        patch_size=16,
        drop_path_rate=0.1,
        final_norm=False),
    head=dict(
        type='MAEFinetuneHead',
        num_classes=1000,
        embed_dim=768,
        label_smooth_val=0.1),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))
