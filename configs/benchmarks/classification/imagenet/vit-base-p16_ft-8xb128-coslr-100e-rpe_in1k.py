_base_ = 'vit-b-p16_ft-8xb128-coslr-100e_in1k.py'

# model
model = dict(backbone=dict(use_window=True))
