# model settings

custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)

model = dict(
    type='MMClsImageClassifierWrapper',
    backbone=dict(
        type='mmcls.SwinTransformer',
        arch='base',
        img_size=192,
        drop_path_rate=0.1,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))
