# model settings
model = dict(
    type='CAE',
    backbone=dict(
        type='CAEViT',
        arch='b',
        patch_size=16,
        init_values=0.1,
        qkv_bias=False),
    neck=dict(
        type='CAENeck',
        patch_size=16,
        embed_dims=768,
        num_heads=12,
        regressor_depth=4,
        decoder_depth=4,
        mlp_ratio=4,
        init_values=0.1,
    ),
    head=dict(
        type='CAEHead', tokenizer_path='cae_ckpt/dalle_encoder.pth', lambd=2),
    base_momentum=0.0)
