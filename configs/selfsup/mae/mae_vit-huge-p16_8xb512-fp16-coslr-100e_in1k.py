_base_ = 'mae_vit-large-p16_8xb512-fp16-coslr-300e_in1k.py'

# pre-train for 100 epochs
train_cfg = dict(max_epochs=100)

# model settings
model = dict(
    backbone=dict(type='MAEViT', arch='h', patch_size=14, mask_ratio=0.75),
    neck=dict(type='MAEPretrainDecoder', embed_dim=1280))