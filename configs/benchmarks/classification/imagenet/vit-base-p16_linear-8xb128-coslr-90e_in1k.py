_base_ = 'vit-small-p16_linear-8xb128-coslr-90e_in1k.py'
# MoCo v3 linear probing setting

model = dict(
    backbone=dict(arch='base', frozen_stages=12, norm_eval=True),
    head=dict(in_channels=768))
