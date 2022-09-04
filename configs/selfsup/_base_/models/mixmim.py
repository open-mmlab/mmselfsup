model = dict(
    type='MixMIM',
    data_preprocessor=dict(  # √
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MixMIMTransformer',
        arch='B',
        drop_rate=0.0,
        drop_path_rate=0.0)
    )
