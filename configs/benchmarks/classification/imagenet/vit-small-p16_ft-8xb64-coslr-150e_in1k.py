_base_ = 'vit-base-p16_ft-8xb64-coslr-150e_in1k.py'
# MoCo v3 fine-tuning setting

# model settings
model = dict(
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
    ))
