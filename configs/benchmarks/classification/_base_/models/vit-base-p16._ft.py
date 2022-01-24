model = dict(
    type='VitClassification',
    backbone=dict(
        type='MAEClsViT',
        arch='b',
        patch_size=16,
        global_pool=True,
        drop_path_rate=0.1,
        final_norm=False),
    head=dict(type='MAEFinetuneHead', num_classes=1000, embed_dim=768),
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=1.0,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=1000)