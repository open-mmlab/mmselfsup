_base_ = 'mocov3_vit-base-p16_16xb256-fp16-coslr-800e_in1k-224.py'

train_dataloader = dict(batch_size=64, num_workers=8)

# model settings
model = dict(
    backbone=dict(arch='large'),  # embed_dim = 768
    neck=dict(in_channels=1024),
)
