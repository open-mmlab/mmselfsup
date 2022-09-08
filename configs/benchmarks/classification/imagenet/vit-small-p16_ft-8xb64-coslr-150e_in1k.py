_base_ = 'vit-base-p16_ft-8xb64-coslr-150e_in1k.py'
# MoCoV3 ViT fine-tuning setting

# model settings
model = dict(
    _delete_=True,
    type='ImageClassifier',
    backbone=dict(
        type='mmselfsup.MoCoV3ViT',
        arch='mocov3-small',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8, num_classes=1000),
        dict(type='CutMix', alpha=1.0, num_classes=1000)
    ]))
