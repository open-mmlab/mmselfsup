# model settings
vqkd_encoder = dict(
    arch='base',
    img_size=224,
    patch_size=16,
    in_channels=3,
    out_indices=-1,
    drop_rate=0.,
    drop_path_rate=0.,
    norm_cfg=dict(type='LN', eps=1e-6),
    final_norm=True,
    with_cls_token=True,
    avg_token=False,
    frozen_stages=-1,
    output_cls_token=False,
    use_abs_pos_emb=True,
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    layer_scale_init_value=0.,
    interpolate_mode='bicubic',
    patch_cfg=dict(),
    layer_cfgs=dict(),
    init_cfg=None)

layer_scale_init_value = 0.1
drop_path_rate = 0.  # 0. for 300 epochs and 0.1 for 1600 epochs.
model = dict(
    type='BEiT',
    backbone=dict(
        type='BEiTViT',
        arch='base',
        patch_size=16,
        out_indices=[-4, -1],
        drop_path_rate=drop_path_rate,
        final_norm=False,
        layer_scale_init_value=layer_scale_init_value,
        init_cfg=[
            dict(type='TruncNormal', std=0.02, layer='Linear'),
            dict(type='TruncNormal', std=0.02, layer='Conv2d'),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    neck=dict(
        type='BEiTV2Neck',
        num_layers=2,
        early_layers=9,
        backbone_arch='base',
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    ),
    head=dict(
        type='BEiTV2Head',
        embed_dims=768,
        num_embed=8192,
        loss=dict(type='BEiTLoss')),
    target_generator=dict(
        type='VQKD',
        encoder_config=vqkd_encoder,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa
            'https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/vqkd_encoder.pth'  # noqa
        )))
