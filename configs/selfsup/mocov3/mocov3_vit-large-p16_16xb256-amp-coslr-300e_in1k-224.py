_base_ = 'mocov3_vit-small-p16_16xb256-fp16-coslr-300e_in1k-224.py'

# model settings
model = dict(
    backbone=dict(arch='large'),  # embed_dim = 768
    neck=dict(in_channels=1024),
)
