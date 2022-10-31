_base_ = 'vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'
# MAE linear probing setting

# model settings
model = dict(
    backbone=dict(type='VisionTransformer', arch='large', frozen_stages=24),
    neck=dict(type='mmselfsup.ClsBatchNormNeck', input_features=1024),
    head=dict(in_channels=1024))
